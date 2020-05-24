import math
import numpy
import random
from PIL import Image

ic = 8
oc = 2

generations = 2048
populations = 256
itterations = 4

ninc = ic+oc #Node innovation number counter
linc = ic*oc #Link innovation number counter

ninc_list = []
linc_list = []

brain = []
enviro = []

fitness = [0 for x in range(populations)]
grouping = [[x for x in range(populations)]]

class environment:

	def __init__(self):

		self.width = 16
		self.height = 16

		self.alive = True
		self.fitness = 0
		self.lifetime = 32

		self.vel = random.choice([[0,1], [1,0], [0,-1], [-1,0]])
		self.pos = [random.randint(4, self.width-5), random.randint(4, self.height-5)]

		self.body = []
		self.body.append([self.pos[0] - 3*self.vel[0], self.pos[1] - 3*self.vel[1]])
		self.body.append([self.pos[0] - 2*self.vel[0], self.pos[1] - 2*self.vel[1]])
		self.body.append([self.pos[0] - 1*self.vel[0], self.pos[1] - 1*self.vel[1]])
		self.body.append(list(self.pos))

		self.food = [random.randint(0, self.width-1), random.randint(0, self.height-1)]
		while self.food in self.body:
			self.food = [random.randint(0, self.width-1), random.randint(0, self.height-1)]

	def update(self, action):

		dst = [0]*ic

		if self.alive:

			# Turn Left (x,y) -> (-y,x)
			if action[0] >= 0.75:
				self.vel = [-self.vel[1], self.vel[0]]

			# Turn Right (x,y) -> (y,-x)
			if action[1] >= 0.75:
				self.vel = [self.vel[1], -self.vel[0]]

			# Calculate next position
			self.pos[0] += self.vel[0]
			self.pos[1] += self.vel[1]

			#If collided with left or right wall
			if self.pos[0] < 0 or self.pos[0] >= self.width:
				self.alive = False
				return dst

			#If collided with top or bottom wall
			if self.pos[1] < 0 or self.pos[1] >= self.height:
				self.alive = False
				return dst

			#If collided with body
			if self.pos in self.body:
				self.alive = False
				return dst

			#If died of old age
			if self.lifetime <= 0:
				self.alive = False
				return dst

			#If collided with food
			if self.pos == self.food:
				self.body.append(list(self.pos))
				self.fitness += 1
				self.lifetime += 16
				while self.food in self.body:
					self.food = [random.randint(0, self.width-1), random.randint(0, self.height-1)]

			# calculate sensor values (manual)

			dst[0] = (self.pos[0] - self.food[0])*self.vel[0] #x diff from food
			dst[1] = (self.pos[1] - self.food[1])*self.vel[1] #y diff from food

			dst[2] = 1 if (0 < (self.pos[0] + self.vel[0]) >= self.width) or (0 < (self.pos[1] + self.vel[1]) >= self.height) else 0 #front wall
			dst[3] = 1 if (0 < (self.pos[0] - self.vel[1]) >= self.width) or (0 < (self.pos[1] + self.vel[0]) >= self.height) else 0 #left wall
			dst[4] = 1 if (0 < (self.pos[0] + self.vel[1]) >= self.width) or (0 < (self.pos[1] - self.vel[0]) >= self.height) else 0 #right wall

			dst[5] = 1 if [self.pos[0] + self.vel[0], self.pos[1] + self.vel[1]] in self.body else 0 #front body
			dst[6] = 1 if [self.pos[0] - self.vel[1], self.pos[1] + self.vel[0]] in self.body else 0 #left body
			dst[7] = 1 if [self.pos[0] + self.vel[1], self.pos[1] - self.vel[0]] in self.body else 0 #right body

			# Move body
			if self.body:
				self.body.pop(0)
			self.body.append(list(self.pos))
			self.fitness -= 0.01 #life is pain
			self.lifetime -= 1 #life is short

		return dst

class network:

	def __init__(self):

		self.nc = ninc
		self.nodes = []
		self.links = []

	def initnew(self):
		global ninc, linc

		#Create nodes (one for every input and output) (node-id, node-sum, node-out)
		for i in range(ic+oc):
			self.nodes.append([i, 0, 0])

		#Create links (one from every input to every output) (link-id, node-src, node-dst, link-weight)
		for i in range(ic*oc):
			self.links.append([i, i//oc, i%oc+ic, random.uniform(-1.0, 1.0)])

		self.nodes.sort(key=lambda a: a[0]) #Sort using first element (node-id)
		self.links.sort(key=lambda a: a[0]) #Sort using first element (link-id)

	def initchild(self, p1, p2, s1, s2):
		global ninc, linc, ninc_list, linc_list

		#Crossover nodes
		
		a = 0
		b = 0

		while a < len(p1.nodes) and b < len(p2.nodes):

			if p1.nodes[a][0] == p2.nodes[b][0]:
				if random.choice([True, False]):
					self.nodes.append(list(p1.nodes[a]))
				else:
					self.nodes.append(list(p2.nodes[b]))
				a += 1
				b += 1
				continue

			if p1.nodes[a][0] < p2.nodes[b][0]:
				if s1 >= s2:
					self.nodes.append(list(p1.nodes[a]))
				a += 1
				continue

			if p1.nodes[a][0] > p2.nodes[b][0]:
				if s1 < s2:
					self.nodes.append(list(p2.nodes[b]))
				b += 1
				continue

		if s1 >= s2:
			while a < len(p1.nodes):
				self.nodes.append(list(p1.nodes[a]))
				a += 1

		if s1 < s2:
			while b < len(p2.nodes):
				self.nodes.append(list(p2.nodes[b]))
				b += 1

		#Crossover links
		a = 0
		b = 0

		while a < len(p1.links) and b < len(p2.links):

			if p1.links[a][0] == p2.links[b][0]:
				if random.choice([True, False]):
					self.links.append(list(p1.links[a]))
				else:
					self.links.append(list(p2.links[b]))

				a += 1
				b += 1
				continue

			if p1.links[a][0] < p2.links[b][0]:
				if s1 >= s2:
					self.links.append(list(p1.links[a]))
				a += 1
				continue

			if p1.links[a][0] > p2.links[b][0]:
				if s1 < s2:
					self.links.append(list(p2.links[b]))
				b += 1
				continue

		if s1 >= s2:
			while a < len(p1.links):
				self.links.append(list(p1.links[a]))
				a += 1

		if s1 < s2:
			while b < len(p2.links):
				self.links.append(list(p2.links[b]))
				b += 1

		#Mutation New node
		if random.random() < 0.01:

			c = self.links.pop(random.randint(0, len(self.links)-1))

			res = [x for x in ninc_list if x[0] == c[0]]
			
			if not res:
				ninc_list.append([c[0],[ninc, 0, 0],[linc, c[1], ninc, 1],[linc+1, ninc, c[2], c[3]]])

				self.nodes.append(ninc_list[-1][1])
				self.links.append(ninc_list[-1][3])
				self.links.append(ninc_list[-1][2])

				ninc += 1
				linc += 1
				linc += 1
			else:
				self.nodes.append(res[0][1])
				self.links.append(res[0][2])
				self.links.append(res[0][3])



		#Mutation New link
		if random.random() < 0.05:

			if len(self.links) < ic*oc:

				a = random.randint(ic+oc, len(self.nodes)-1)
				b = random.randint(ic, len(self.nodes)-1)

				aaa = self.nodes[a][0]
				bbb = self.nodes[b][0]

				res = [x for x in linc_list if (x[1] == aaa and x[2] == bbb)]
				res2 = [x for x in self.links if (x[1] == aaa and x[2] == bbb)]

				if not res and not res2:
					res = [linc, aaa, bbb, random.uniform(-1,1)]
					linc_list.append(res)
					linc += 1
				else:
					res = res[0]
				
				self.links.append(res)
			

		#Mutation Old link
		if random.random() < 0.1:
			self.links[random.randint(0,len(self.links)-1)][3] = random.uniform(-1,1)

		if random.random() < 0.8:
			for i in range(len(self.links)):
				self.links[i][3] += random.gauss(0, 1) / 50
				self.links[i][3] = min(1, self.links[i][3])
				self.links[i][3] = max(-1, self.links[i][3])



		#Mutation Remove link
		#if random.random() < 0.05:
		#	if(len(self.links) > (ic+oc)):
		#		self.links.pop(random.randint(ic+oc,len(self.links)-1))

		self.nodes.sort(key=lambda a: a[0]) #Sort using first element (node-id)
		self.links.sort(key=lambda a: a[0]) #Sort using first element (link-id)



	def update(self, stimulus):

		for i in range(ic):
			self.nodes[i][1] = stimulus[i]

		for i in range(len(self.nodes)):
			self.nodes[i][2] = 1/(1+math.exp(-4.9*self.nodes[i][1])) #Activation (sigmoid)
			#self.nodes[i][2] = max(0, self.nodes[i][1]) #Activation (relu)
			#self.nodes[i][2] = self.nodes[i][1] / (abs(self.nodes[i][1]) + 1)
			self.nodes[i][1] = 0

		a = 0
		b = 0
		for i in range(len(self.links)):

			while self.nodes[a][0] < self.links[i][1]:
				a += 1
			while self.nodes[a][0] > self.links[i][1]:
				a -= 1
			while self.nodes[b][0] < self.links[i][2]:
				b += 1
			while self.nodes[b][0] > self.links[i][2]:
				b -= 1

			self.nodes[b][1] += self.nodes[a][2] * self.links[i][3] #Propagation

		return [x[2] for x in self.nodes[ic:ic+oc]]



for i in range(populations):
	brain.append(network())
	enviro.append(environment())
	brain[i].initnew()

print("Initialized")

for gen in range(generations):

	for pop in range(populations):

		stimuli = [0]*ic
		action = [0]*oc

		while enviro[pop].alive:

			stimuli = enviro[pop].update(action)

			for itt in range(itterations):

				action = brain[pop].update(stimuli)

		fitness[pop] = enviro[pop].fitness
		enviro[pop] = environment()

	print("Generation: "+str(gen))

	print("Node inovation number: "+str(ninc))
	print("Link inovation number: "+str(linc))
	'''
	print("Nodesize-min: "+str(min([len(a.nodes) for a in brain])))
	print("Nodesize-max: "+str(max([len(a.nodes) for a in brain])))
	print("Nodesize-avg: "+str(sum([len(a.nodes) for a in brain])/len(brain)))

	print("Linksize-min: "+str(min([len(a.links) for a in brain])))
	print("Linksize-max: "+str(max([len(a.links) for a in brain])))
	print("Linksize-avg: "+str(sum([len(a.links) for a in brain])/len(brain)))
	'''
	print("Fitness-min: "+str(min(fitness)))
	print("Fitness-max: "+str(max(fitness)))
	print("Fitness-avg: "+str(sum(fitness)/len(fitness)))

	#####################
	# Save animated gif #
	#####################

	if gen%10 == 0:
		images = []
		tmp_env = environment()
		tmp_net = brain[numpy.argsort(numpy.asarray(fitness))[-1]]

		tw = 256
		th = 256
		dw = tw/tmp_env.width
		dh = th/tmp_env.height

		stimuli = [0]*ic
		action = [0]*oc

		while tmp_env.alive:
			imgarray = numpy.zeros((tw, th, 3), dtype=numpy.uint8)
			imgarray[int(tmp_env.food[0]*dw):int(tmp_env.food[0]*dw+dw-1),int(tmp_env.food[1]*dh):int(tmp_env.food[1]*dh+dh-1)] = [0,255,0]
			for seg in tmp_env.body:
				imgarray[int(seg[0]*dw):int(seg[0]*dw+dw-1),int(seg[1]*dh):int(seg[1]*dh+dh-1)] = [255,255,255]
			images.append(Image.fromarray(numpy.array(imgarray), mode="RGB"))

			stimuli = tmp_env.update(action)
			for itt in range(itterations):
				action = tmp_net.update(stimuli)

		images[0].save(str(gen)+'.gif', save_all=True, append_images=images[1:], optimize=True, duration=64, loop=0)

	##########################################
	# Speciation (group based on similarity) #
	##########################################

	#Group the networks into groups based on similarity. (this code is uglu AF)
	grand = [] #rand element in group
	gtemp = [] #next version of grouping

	for group in range(len(grouping)):

		gtemp.append([])
		grand.append(random.choice(grouping[group]))

	for pop in range(populations):

		doesnotfitingroup = True

		#what group does this pop currenlty reside in

		popgr = 0
		for gg in range(len(grouping)):
			if pop in grouping[gg]:
				popgr = gg

		for gg in range(len(grand)):
			gro = (gg + popgr) % len(grand) #start with parent group
			gro = 0

			ldif = 0
			wdif = 0
			wcunt = 0


			a = 0
			b = 0

			while a < len(brain[grand[gro]].links) and b < len(brain[pop].links):

				if brain[grand[gro]].links[a][0] < brain[pop].links[b][0]:
					ldif += 1
					a += 1
					continue

				if brain[grand[gro]].links[a][0] > brain[pop].links[b][0]:
					ldif += 1
					b += 1
					continue
				
				wdif += abs(brain[grand[gro]].links[a][3] - brain[pop].links[b][3])
				wcunt += 1

				a += 1
				b += 1

			while a < len(brain[grand[gro]].links):
				ldif += 1
				a += 1

			while b < len(brain[pop].links):
				ldif += 1
				b += 1

			difftot = ldif/max(len(brain[grand[gro]].links),len(brain[pop].links))+wdif/max(1, wcunt)
			
			if difftot < 1: #put in group where dissimilarity less than 4

				gtemp[gro].append(pop)
				doesnotfitingroup = False
				break

		if doesnotfitingroup:

			gtemp.append([pop])
			grand.append(pop)

	grouping = gtemp
	grouprev = [0]*populations

	#grand = [random.choice(x) for x in grouping]
	#gtemp = [[] for x in range(len(grouping))]

	#grouping = [[x for x in range(populations)]]
	#grouprev = [0]*populations

	print("Grouping: "+str([len(x) for x in grouping]))
	#print("Grouping: "+str([x for x in grouping]))

	#Fitness sharing
	for group in range(len(grouping)):
		for node in range(len(grouping[group])):
			fitness[node] /= len(grouping[group])
			grouprev[node] = group
	
	'''
	print("Fitness-shared-min: "+str(min(fitness)))
	print("Fitness-shared-max: "+str(max(fitness)))
	print("Fitness-shared-avg: "+str(sum(fitness)/len(fitness)))
	'''

	#Get list of highest scoring indexes
	score_top = numpy.argsort(numpy.asarray(fitness))[-int(len(fitness)/8):]

	'''
	print("Fittest-nodes: "+str(len(brain[score_top[-1]].nodes)))
	print("Fittest-links: "+str(len(brain[score_top[-1]].links)))
	'''

	#Remove lowest scoring indexes from groups
	grouping = [list(set(x) & set(score_top)) for x in grouping]

	for pop in range(populations):
		
		if pop not in score_top:

			i1 = numpy.random.choice(score_top) # Pick any network
			i2 = numpy.random.choice(score_top) #numpy.random.choice(grouping[grouprev[i1]]) #Pick any network in the same group

			grouping[random.choice([grouprev[i1], grouprev[i2]])].append(pop) #Child group assignment

			brain[pop] = network()
			brain[pop].initchild(brain[i1], brain[i2], fitness[i1], fitness[i2])

	#Remove empty lists in grouping
	grouping = [x for x in grouping if x]
	print()

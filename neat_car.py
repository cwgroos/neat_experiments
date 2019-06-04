from __future__ import print_function
import gym
import os
import neat
env = gym.make('MountainCarContinuous-v0')

def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
          try:
            state = env.reset()
            print(genome)
	    net = neat.nn.FeedForwardNetwork.create(genome, config)
            #reward = 0
            done = False
	    while not done:
              output = net.activate(state)
              #print(output)
              #print(state)
	      s, reward, done, info = env.step(output)
              state = s
	      
	      #print(reward)
	      genome.fitness = reward
              if genome.fitness >= 80:
		print(state)
		print(genome_id)
		print(genome.fitness)
		#exit()
	  except KeyboardInterrupt:
            env.close()
            exit()
	  print(genome.fitness)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.getcwd(), 'config')
    print(config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    gen_best = pop.run(eval_genomes, 500)
    #print(gen_best.id)
    best_genome = stats.best_genome()
    print(best_genome)
    observation = env.reset()
    net = neat.nn.FeedForwardNetwork.create(best_genome, config) 
    done = False
    while not done:
      env.render()
      action = net.activate(observation)
      state, reward, done, info = env.step(action)
      observation = state
    print(reward)
    print(state)
    print(done)
    env.close()

if __name__ == "__main__":
        #env.reset()    
        #state, reward, done, info = env.step(env.action_space.sample())
        #print(state.flatten().shape)
        run()

# Copyright 1996-2021 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from controller import Supervisor

from utilities import normalizeToRange

try:
	import gym
	import numpy as np
	from stable_baselines3 import PPO
	from stable_baselines3.common.env_checker import check_env
except ImportError:
	sys.exit(
		'Please make sure you have all dependencies installed. '
		'Run: "pip3 install numpy gym stable_baselines3"'
	)


class OpenAIGymEnvironment(Supervisor, gym.Env):
	def __init__(self, max_episode_steps=1000):
		super().__init__()

		# Open AI Gym generic
		high = np.array([1,1,1,1],dtype=np.float32
		)
		self.action_space = gym.spaces.Discrete(3) # 3 actions Left, Right, go Straight
		self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
		self.state = None
		self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

		# Environment specific
		self.__timestep = int(self.getBasicTimeStep())


		# Robot
		self.__wheels = []
		self.__camera = self.getDevice('camera')
		self.__camera.enable(self.__timestep)
		self.__camera.recognitionEnable(self.__timestep)


		# Rubber Duckie
		self.__duckie = self.getFromDef('rubber_duck')
		self.__duckie_Transfield = self.__duckie.getField('translation')


		# Tools
		self.keyboard = self.getKeyboard()
		self.keyboard.enable(self.__timestep)

		print("initialized Robot and Environment")

		self.step_count = 0



	def wait_keyboard(self):
		while self.keyboard.getKey() != ord('Y'):
			super().step(self.__timestep)

	def reset(self):
		# Reset the simulation
		self.simulationResetPhysics()
		self.simulationReset()
		super().step(self.__timestep)

		# Motors
		self.__wheels = []
		for name in ['left_wheel_hinge', 'right_wheel_hinge']:
			wheel = self.getDevice(name)
			wheel.setPosition(float('inf'))
			wheel.setVelocity(0)
			self.__wheels.append(wheel)

		# reset random Rubber Duckie postion
		newDuckY_Position = np.random.uniform(low=-1, high=1)
		duckX_position = 0. # fix x position
		duckZ_position = 0.023 # fix the elevation position, move only along Y axis
		self.__duckie_Transfield.setSFVec3f([duckX_position,newDuckY_Position,duckZ_position])


		# Internals
		super().step(self.__timestep)

		# Open AI Gym generic
		# return np.array([0, 0, 0, 0]).astype(np.float32)
		return np.array([0.0 for _ in range(self.observation_space.shape[0])]).astype(np.float32)

	def step(self, action):

		self.step_count+=1

		# Execute the action

		if action == 0:
			# go left
			self.__wheels[0].setVelocity(4)
			self.__wheels[1].setVelocity(6)
		elif action == 1:
			# go straight
			self.__wheels[0].setVelocity(6)
			self.__wheels[1].setVelocity(6)
		else:
			# go right
			self.__wheels[0].setVelocity(6)
			self.__wheels[1].setVelocity(4)
		
		
		super().step(self.__timestep)

		# Observation
		robot = self.getSelf()
		objectDetected = self.__camera.getRecognitionObjects()
		duckieImagePosX = 1.0
		duckieImagePosY = 1.0
		duckieImageSize = 0.0
		for obj in objectDetected:
			if obj.model == 'rubber duck':
				duckieImagePosX = normalizeToRange(obj.get_position_on_image()[0], 0, 224, -1, 1, clip=False)
				duckieImagePosY = normalizeToRange(obj.get_position_on_image()[1], 0, 224, -1, 1, clip=False)
				duckieImageSize = obj.get_size_on_image()[0]*obj.get_size_on_image()[1]/(224*224)

		# Camera Obs
		# return [duckieImageSize, duckieImageSize, duckieImagePosX, duckieImagePosY]
		self.state = np.array([duckieImageSize, duckieImageSize, duckieImagePosX, duckieImagePosY])

		# Done
		done = bool(self.step_count==250)
		if done:
			self.step_count = 0

		# Reward
		robot_Position = robot.getField('translation').getSFVec3f()
		duck_Position = self.__duckie_Transfield.getSFVec3f()
		distance = np.sqrt((robot_Position[0]-duck_Position[0])**2  + (robot_Position[1]-duck_Position[1])**2)
		reward = 1/distance

		return self.state.astype(np.float32), reward, done, {}


def main():
	# Initialize the environment
	env = OpenAIGymEnvironment()
	check_env(env)

	# Train
	model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)
	model.learn(total_timesteps=500000)

	# Replay
	# print('Training is finished, press `Y` for replay...')
	# env.wait_keyboard()

	obs = env.reset()
	for _ in range(100000):
		action, _states = model.predict(obs)
		obs, reward, done, info = env.step(action)
		print(obs, reward, done, info)
		if done:
			obs = env.reset()


if __name__ == '__main__':
	main()

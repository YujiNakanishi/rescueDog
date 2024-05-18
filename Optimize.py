'''
/******************************************/
module : rescueDog version 1.0
File name : Optimize.py
Author : Yuji Nakanishi
Latest update : 2019/12/24
/******************************************/

Class list:
Function list:
'''

from myModule import rescueDog as rd
from myModule import pyMath
import numpy as np
import sys

def checkConverge(fitness_before, fitness_after, epsilon):
	diff = np.abs(fitness_after - fitness_before)

	return (diff < epsilon)

class Base:
	def __init__(self, h = 1e-5, epsilon = 0.):
		self.h = h
		self.epsilon = epsilon
		self.fitness_before = None
		self.fitness_after = None

'''
/******************/
GradDescent
/******************/
type : class
gradient descentによる最適値探査
Field : 
Method : progress
'''
class GradDescent(Base):

	'''
	/*******************/
	progress
	/*******************/
	type : method
	process : 最急降下法による解の更新
	Input : solution, evaluation, alpha
		solution -> solutionクラス
		evaluation -> evaluationクラス
		alpha -> float。更新率。緩和係数。
	Output : unconverged_solution -> solutionクラス。
	Note
		:名前のせいで誤解を招くかもしれないが、unconverged_solutionにはconveregedなsolutionも(最終的には)含まれている。
	'''
	def progress(self, solution, evaluation, alpha = 1.):
		#####unconvergedなsolutionのみ処理を施す．
		converged_solution, unconverged_solution = rd.Population.Separate(solution)

		if unconverged_solution is None:
			print("All points have been counverged")
			return solution
		else:
			self.fitness_before = evaluation.getFitness(unconverged_solution) #(Nu, )なshape

			Gradient = rd.Population.Gradient(unconverged_solution, evaluation.function, self.h)

			#####unconverged_solutionの更新
			if rd._optimize == "minimize":
				unconverged_solution.points -= alpha*Gradient
			else:
				unconverged_solution.points += alpha*Gradient

			self.fitness_after = evaluation.getFitness(unconverged_solution) #(Nu, )なshape

			#####unconverged_solution.convergeの更新
			unconverged_solution.converge = checkConverge(self.fitness_before, self.fitness_after, self.epsilon) #(Nu, )なshape

			if converged_solution is None:
				return unconverged_solution
			else:
				unconverged_solution += converged_solution
				return unconverged_solution

'''
/******************/
Newton_Raphson
/******************/
type : class
gradient ニュートン法による最適値探査
Field : 
Method : progress
'''
class Newton_Raphson(Base):

	'''
	/*******************/
	progress
	/*******************/
	type : method
	process : 最急降下法による解の更新
	Input : solution, evaluation, alpha
		solution -> solutionクラス
		evaluation -> evaluationクラス
		alpha -> float。更新率。緩和係数。
	Output : unconverged_solution -> solutionクラス。
	Note
		:名前のせいで誤解を招くかもしれないが、unconverged_solutionにはconveregedなsolutionも(最終的には)含まれている。
	'''
	def progress(self, solution, evaluation, alpha = 1.):
		#####unconvergedなsolutionのみ処理を施す．
		converged_solution, unconverged_solution = rd.Population.Separate(solution)

		if unconverged_solution is None:
			print("All points have been counverged")
			return solution
		else:
			self.fitness_before = evaluation.getFitness(unconverged_solution) #(Nu, )なshape

			#####進行方向の計算
			Direction = None #最終的には(Nu, D)なshape

			for point in unconverged_solution:
				#####pointは(D, )なshape

				#####ヘッセ行列の計算
				H = pyMath.Analysis.HessianMatrix(point, evaluation.function, self.h) #(D, D)なshape
				direction = alpha*np.dot(np.linalg.inv(H), pyMath.Analysis.Gradient(evaluation.function, point, self.h))
				direction = np.expand_dims(direction, axis = 0) #(1, D)なshape

				if Direction is None:
					Direction = direction
				else:
					Direction = np.concatenate((Direction, direction), axis = 0)

			#####unconverged_solutionの更新
			if rd._optimize == "minimize":
				unconverged_solution.points -= alpha*Direction
			else:
				unconverged_solution.points += alpha*Direction

			self.fitness_after = evaluation.getFitness(unconverged_solution) #(Nu, )なshape

			#####unconverged_solution.convergeの更新
			unconverged_solution.converge = checkConverge(self.fitness_before, self.fitness_after, self.epsilon) #(Nu, )なshape

			if converged_solution is None:
				return unconverged_solution
			else:
				unconverged_solution += converged_solution
				return unconverged_solution


'''
/******************/
Conjugate_Gradient
/******************/
type : class
gradient 共役勾配法による最適値探査
Field : d, beta, way
	d -> np配列。(N, D)なshape。更新方向
	beta -> float。共役勾配法で用いられる値。
	way -> betaの計算方法。
Method : progress
'''
class Conjugate_Gradient:

	def __init__(self, h = 1e-5, epsilon = 0., way = "Polak"):
		self.h = h
		self.epsilon = epsilon
		self.way = way
		self.fitness_before = None
		self.fitness_after = None
		self.d = None
		self.beta = None

	'''
	/*****************/
	Polak
	/*****************/
	type : method
	process : Polak-Ribiereによるbetaの計算方法
	Input : unconverged_solution, unconverged_solution_after, evaluation
		unconverged_solution -> solutionクラス。更新前
		unconverged_solution_after -> solutionクラス。更新後
		evaluation -> evaluationクラス
	Output : なし。self.bateを更新。
	'''
	def Polak(self, unconverged_solution, unconverged_solution_after, evaluation):
		gradient = rd.Population.Gradient(unconverged_solution, evaluation.function, self.h)
		gradient_after = rd.Population.Gradient(unconverged_solution_after, evaluation.function, self.h)

		y = gradient_after - gradient

		self.beta = np.dot(gradient_after, y.T) / np.dot(gradient, gradient.T)

	'''
	/*******************/
	progress
	/*******************/
	type : method
	process : 最急降下法による解の更新
	Input : solution, evaluation, alpha
		solution -> solutionクラス
		evaluation -> evaluationクラス
		alpha -> float。更新率。緩和係数。
	Output : unconverged_solution -> solutionクラス。
	Note
		:名前のせいで誤解を招くかもしれないが、unconverged_solutionにはconveregedなsolutionも(最終的には)含まれている。
	'''
	def progress(self, solution, evaluation, alpha = 1.):
		#####unconvergedなsolutionのみ処理を施す．
		converged_solution, unconverged_solution = rd.Population.Separate(solution)

		if unconverged_solution is None:
			print("All points have been counverged")
			return solution

		else:
			self.fitness_before = evaluation.getFitness(unconverged_solution) #(Nu, )なshape

			if self.d is None:
				self.d = rd.Population.Gradient(solution, evaluation.function, self.h) #(N, D)なshape
			else:
				self.d[solution.converge] = np.zeros(solution.points.shape[1])

			d_interest = self.d[solution.converge == False]

			unconverged_solution_after = rd.Population.Copy(unconverged_solution)
			#####unconverged_solutionの更新
			if rd._optimize == "minimize":
				unconverged_solution_after.points += alpha*d_interest
			else:
				unconverged_solution_after.points -= alpha*d_interest

			#####betaの更新
			if self.way == "Polak":
				self.Polak(unconverged_solution, unconverged_solution_after, evaluation)
			else:
				print("Error@rescueDog.Optimize.Conjugate_Gradient.progress")
				print("param way you set is not defined")
				sys.exit()

			#####dの更新
			gradient_after = rd.Population.Gradient(unconverged_solution_after, evaluation.function, self.h)
			self.d[solution.converge == False] = -gradient_after + self.beta*d_interest

			self.fitness_after = evaluation.getFitness(unconverged_solution_after) #(Nu, )なshape

			#####unconverged_solution.convergeの更新
			unconverged_solution_after.converge = checkConverge(self.fitness_before, self.fitness_after, self.epsilon) #(Nu, )なshape

			if converged_solution is None:
				return unconverged_solution_after
			else:
				unconverged_solution_after += converged_solution
				return unconverged_solution_after

class quasi_Newton:

	def __init__(self, h = 1e-5, epsilon = 0., way = "BFGS"):
		self.h = h
		self.epsilon = epsilon
		self.way = way
		self.fitness_before = None
		self.fitness_after = None
		self.B = None

	def BFGS(self, unconverged_solution, unconverged_solution_after, function):
		Nu, D = unconverged_solution.points.shape
		S = unconverged_solution_after.points - unconverged_solution.points #(Nu, D)なshape
		
		gradient = rd.Population.Gradient(unconverged_solution, function, self.h)
		gradient_after = rd.Population.Gradient(unconverged_solution_after, function, self.h)
		Y = gradient_after - gradient #(Nu, D)なshape

		new_B = None

		for B, y, s in zip(self.B, Y, S):
			s = s.reshape((D, 1)); y = s.reshape((D, 1))
			B = B - np.dot(B@s, s.T@B)/np.dot(s.T, B@s) + y@y.T/(s.T@y) #(D, D)なshape
			B = np.expand_dims(B, axis = 0)

			if new_B is None:
				new_B = B
			else:
				new_B = np.concatenate((new_B, B), axis = 0)

		return new_B

	'''
	/*******************/
	progress
	/*******************/
	type : method
	process : 最急降下法による解の更新
	Input : solution, evaluation, alpha
		solution -> solutionクラス
		evaluation -> evaluationクラス
		alpha -> float。更新率。緩和係数。
	Output : unconverged_solution -> solutionクラス。
	Note
		:名前のせいで誤解を招くかもしれないが、unconverged_solutionにはconveregedなsolutionも(最終的には)含まれている。
	'''
	def progress(self, solution, evaluation, alpha = 1.):
		#####unconvergedなsolutionのみ処理を施す．
		converged_solution, unconverged_solution = rd.Population.Separate(solution)

		if unconverged_solution is None:
			print("All points have been counverged")
			return solution

		else:
			self.fitness_before = evaluation.getFitness(unconverged_solution) #(Nu, )なshape

			if self.B is None:
				for i in range(len(solution)):
					I = np.expand_dims(np.eye(solution.points.shape[1]), axis = 0)
					if self.B is None:
						self.B = I
					else:
						self.B = np.concatenate((self.B, I), axis = 0)
			else:
				self.B = self.B[solution.converge == False] #(Nu, D, D)なshape

			gradient = rd.Population.Gradient(unconverged_solution, evaluation.function, self.h) #(Nu, D)なshape

			direction = None

			for B, grad in zip(self.B, gradient):
				B_inv = np.linalg.inv(B)
				d = -np.expand_dims(np.dot(B_inv, grad), axis = 0)

				if direction is None:
					direction = d
				else:
					direction = np.concatenate((direction, d), axis = 0) #(Nu, D)なshape

			unconverged_solution_after = rd.Population.Copy(unconverged_solution)
			#####unconverged_solutionの更新
			if rd._optimize == "minimize":
				unconverged_solution_after.points += alpha*direction
			else:
				unconverged_solution_after.points -= alpha*direction

			if self.way == "BFGS":
				self.B = self.BFGS(unconverged_solution, unconverged_solution_after, evaluation.function)
			else:
				print("Error@rescueDog.Optimize.Conjugate_Gradient.progress")
				print("param way you set is not defined")
				sys.exit()

			self.fitness_after = evaluation.getFitness(unconverged_solution_after) #(Nu, )なshape

			#####unconverged_solution.convergeの更新
			unconverged_solution_after.converge = checkConverge(self.fitness_before, self.fitness_after, self.epsilon) #(Nu, )なshape

			if converged_solution is None:
				return unconverged_solution_after
			else:
				unconverged_solution_after += converged_solution
				return unconverged_solution_after
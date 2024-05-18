'''
/******************************************/
module : rescueDog version 1.0
File name : Fitness.py
Author : Yuji Nakanishi
Latest update : 2019/12/24
/******************************************/

Class list:
	Evaluation
Function list:
	getViolation
'''

from myModule import rescueDog as rd
import numpy as np


'''
/*******************/
getViolation
/*******************/
type : function
Input : solution, penalties
	solution -> solutionクラス
	penalties -> list
		factor -> penaltyクラス
Output : np配列．(N, )なshape
'''
def getViolation(solution, penalties):

	if penalties is None:
		return np.zeros(len(solution))
	else:
		violations = None
		for point in solution:
			violation = 0.
			for penalty in penalties:
				violation += penalty.Violation(point)

			if violations is None:
				violations = np.array([violation])
			else:
				violations = np.concatenate((violations, np.array([violation])))

		if rd._optimize == "minimize":
			return violations
		else:
			return -violations

'''
/******************/
Evaluation
/******************/
type : class
Field : function, penalties
	function -> func．
		Input -> (D, )なshape．
		Output -> float
	penalties -> list
		factor -> penaltyクラス
Method : getFitness, getElite
'''
class Evaluation:

	def __init__(self, function, penalties = None):
		self.function = function
		self.penalties = penalties

	'''
	/*****************/
	getFitness
	/*****************/
	type : function
	process : fitness値を計算する
	Input : solution -> solutionクラス
	Output : np配列．(N, )なshape．
	'''
	def getFitness(self, solution):
		evaluation = None #最終的にはfitness値になる．

		for point in solution:
			if evaluation is None:
				evaluation = np.array([self.function(point)])
			else:
				evaluation = np.concatenate((evaluation, np.array([self.function(point)])))

		evaluation += getViolation(solution, self.penalties)


		return evaluation

	'''
	/*******************/
	getElite
	/*******************/
	type : function
	Input : solution, k
		solution -> solutionクラス
		k -> eliteの数
	Output : elite．solutionクラス．eliteの点の集合．
	'''
	def getElite(self, solution, k = 1):
		
		solution_copy = rd.Population.Copy(solution)
		if not(self.penalties is None):
			solution_copy = rd.Penalty.Feasibles(solution_copy, self.penalties)

		fitness = self.getFitness(solution_copy) #(N, )なshape

		if rd._optimize == "minimize":
			sort_index = np.argsort(fitness)
		else:
			sort_index = -np.argsort(fitness)

		elite_index = sort_index[:k]

		elite = solution_copy.points[elite_index]

		return type(solution_copy)(elite)
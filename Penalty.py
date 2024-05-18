'''
/******************************************/
module : rescueDog version 1.0
File name : Penalty.py
Author : Yuji Nakanishi
Latest update : 2019/12/24
/******************************************/

Class list:
	Base, Equal, Lower, Upper
Function list:
	Feasibles
'''

import numpy as np
from myModule import rescueDog as rd

'''
/*************/
Feasibles
/*************/
type : function
process : solutionの中から制約を満たす点を返す
Input : solution, penalties
Output : None or solutionクラス
			制約を満たす点がない -> None
			ある -> solutionクラス
'''
def Feasibles(solution, penalties):
	judge = np.ones(len(solution)).astype(bool)

	for penalty in penalties:
		judge *= penalty.Feasible(solution)

		if np.all(judge == False):
			return None
		else:
			points = solution.points[judge]
			return type(solution)(points)

'''
/************/
Equal
/************/
type : class
親クラス
Field : constraint, weight
	constraint -> function
		Input -> np配列．(D, )なshape
		Output -> float．
	weight -> float．重み．
Method : Feasible
'''
class Base:
	def __init__(self, constraint, weight = 1.):
		self.constraint = constraint
		self.weight = weight

	'''
	/*****************/
	Feasible
	/*****************/
	type : method
	process : 制約を満たすか否か確認する．
	Input : solution -> solutionクラス
	Output : feasible -> np配列．(N, )なshape 
				factor -> bool
	'''
	def Feasible(self, solution):

		feasible = None

		for point in solution.points:
			if feasible is None:
				feasible = np.array([(self.Violation(point) == 0.)])
			else:
				feasible = np.concatenate((feasible, np.array([(self.Violation(point) == 0.)])))

		return feasible


'''
/************/
Equal
/************/
type : class
等式制約
Field : 
Method : Violation
'''
class Equal(Base):

	'''
	/*****************/
	Violation
	/*****************/
	type : method
	process : ペナルティ値を計算する．
	Input : point -> np配列．(D, )なshape
	Output : float．ペナルティ値
	'''
	def Violation(self, point):
		violation = self.constraint(point) #float
		return self.weight*abs(violation)

'''
/************/
Lower
/************/
type : class
制約式 < 0な制約条件
Field : 
Method : Violation
'''
class Lower(Base):

	'''
	/*****************/
	Violation
	/*****************/
	type : method
	process : ペナルティ値を計算する．
	Input : point -> np配列．(D, )なshape
	Output : float．ペナルティ値
	'''
	def Violation(self, point):
		violation = self.constraint(point) #float

		if violation < 0.:
			return 0.
		else:
			return self.weight*violation

'''
/************/
Upper
/************/
type : class
制約式 > 0な制約条件
Field : 
Method : Violation
'''
class Upper(Base):

	'''
	/*****************/
	Violation
	/*****************/
	type : method
	process : ペナルティ値を計算する．
	Input : point -> np配列．(D, )なshape
	Output : float．ペナルティ値
	'''
	def Violation(self, point):
		violation = self.constraint(point) #float

		if violation > 0.:
			return 0.
		else:
			return -self.weight*violation
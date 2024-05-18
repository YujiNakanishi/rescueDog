'''
/******************************************/
module : rescueDog version 1.0
File name : Population.py
Author : Yuji Nakanishi
Latest update : 2019/12/24
/******************************************/

Class list:
	Solution
Function list:
	Copy, Separate
'''

import numpy as np
from myModule import rescueDog as rd
from myModule import pyMath

'''
/****************/
Solution
/****************/
type : class
探査点群を表すクラス
Field : points, converge
'''
class Solution:
	def __init__(self, points, converge = None):
		if len(points.shape) != 2:
			print("Error@rescueDog.Population.Solution.__init__")
			print("dim of points must be 2")
			sys.exit()
		self.points = points

		if converge is None:
			self.converge = np.zeros(points.shape[0]).astype(bool)
		else:
			self.converge = converge

	def __len__(self):
		return self.points.shape[0]

	def __str__(self):
		print(self.points)
		return ""

	def __getitem__(self, index):
		return self.points[index]

	def __add__(self, other):
		
		self_points = np.copy(self.points)
		self_converge = np.copy(self.converge)
		other_points = np.copy(other.points)
		other_converge = np.copy(other.converge)

		new_points = np.concatenate((self_points, other_points), axis = 0)
		new_converge = np.concatenate((self_converge, other_converge), axis = 0)

		return type(self)(new_points, new_converge)

	'''
	/****************/
	checkConverge
	/****************/
	type : method
	process : solutionの全点が収束済みか否かチェックする．
	Output : bool
	'''
	def checkConverge(self):
		return np.all(self.converge)

	'''
	/********************/
	fuse
	/********************/
	process : distance以下の点どうしを一つにする (一つにされた後の点は重心に位置する)．
	Input : distance -> float．閾値(ユークリッド距離)
	Output : solutionクラス．(Nf, D)なshape
	Note
		:全ての点はconverge = Falseに設定する．
	'''
	def fuse(self, distance):
		candidate = np.copy(self.points) #(N, D)なshape
		new_points = None #最終的には(Nf, D)なshapeになる．

		#####全ての点に対し処理を施すまで
		while candidate.shape[0] != 0:
			#####candidate[0]の最近傍点を探す．
			neighber = pyMath.Geometry.Neighber(candidate, 0, 1)[0] #(D, )なshape
			#####最近傍点とのユークリッド距離を求める．
			L2norm = np.linalg.norm(neighber - candidate[0], ord=2) #L2norm

			if L2norm < distance:
				######2点の重心を求める．
				new_point = (neighber + candidate[0])/2.
				new_point = np.expand_dims(new_point, axis = 0) #(1, D)なshape
			else:
				#####近傍に点がないということで，candidate[0]は保存．
				new_point = candidate[0]
				new_point = np.expand_dims(new_point, axis = 0) #(1, D)なshape

			if new_points is None:
				new_points = new_point
			else:
				new_points = np.concatenate((new_points, new_point))

			#####neighber点のindexを求める．
			neighber_idx = np.all((candidate == neighber), axis = 1) #(N, )なshape.
			#####candidateからneighber点を除く．
			candidate = candidate[neighber_idx == False]
			#####candidate[0]を除く．
			candidate = candidate[1:]

		num_points = new_points.shape[0]
		new_converge = np.zeros(num_points).astype(bool)

		return type(self)(new_points, new_converge)



'''
/****************/
Copy
/****************/
type : function
process : solutionのコピーを作成
Input : solution -> solutionクラス
Output : solutionクラス
'''
def Copy(solution):
	points = np.copy(solution.points)
	converge = np.copy(solution.converge)
	return type(solution)(points, converge)


'''
/****************/
Separate
/****************/
type : function
process : solutionを収束済みの点とそうでないものに分ける
Input : solution -> solutionクラス
Output : converge_solution, unconverge_solution -> solutionクラス
Note
	:収束済み，もしくはそうでない点がない場合，Noneを返す．
'''
def Separate(solution):
	converge_points = np.copy(solution.points[solution.converge])
	if len(converge_points) == 0:
		converge_solution = None
	else:
		num_converge = len(converge_points)
		converge_solution = type(solution)(converge_points, np.ones(num_converge).astype(bool))

	unconverge_points = np.copy(solution.points[solution.converge == False])
	if len(unconverge_points) == 0:
		unconverge_solution = None
	else:
		num_unconverge = len(unconverge_points)
		unconverge_solution = type(solution)(unconverge_points, np.zeros(num_unconverge).astype(bool))

	return converge_solution, unconverge_solution

'''
/****************/
Gradient
/****************/
type : function
process : solutionに対して勾配を計算する。
Input : solution, function, h
	solution -> solutionクラス
	function -> func．
		Input -> (D, )なshape．
		Output -> float
	h -> float。step幅
		None -> pyMath.Analysis.Gradientのデフォルト引数。
Output : gradient -> np配列。(N, D)なshape。勾配。．
'''
def Gradient(solution, function, h = 1e-5):
	gradient = None

	for point in solution:
		grad = pyMath.Analysis.Gradient(function, point, h)
		grad = np.expand_dims(grad, axis = 0) #(1, D)なshape

		if gradient is None:
			gradient = grad
		else:
			gradient = np.concatenate((gradient, grad), axis = 0)

	return gradient
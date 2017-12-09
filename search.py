# coding=utf-8
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import math
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    search_stack = util.Stack()

    successors = problem.getSuccessors(problem.getStartState())

    for ea in successors:
        search_stack.push(ea)

    find_goal = False

    path_actions = []

    visited_pos = set()
    visited_pos.add(problem.getStartState())


# 开始遍历：如果栈不空，且没有到达目标结点(请填充如下两个条件)：
    while not search_stack.isEmpty() and find_goal == False:
        choice = search_stack.pop()
        if not problem.isGoalState(choice[0]):
# 如果该节点没被访问
            if choice[0] not in visited_pos:
                visited_pos.add(choice[0])
                path_actions.append(choice)
    # filter的意思是对sequence中的所有item依次执行 function(item)
            choice_successors = filter(lambda v: v[0] not in visited_pos, problem.getSuccessors(choice[0]))

            if not len(choice_successors):
                path_actions.pop(-1)
                if path_actions:
                    search_stack.push(path_actions[-1])
            else:
                for ea in choice_successors:
                    search_stack.push(ea)

        else:
            path_actions.append(choice)
            visited_pos.add(choice[0])
            find_goal = True


    return [ea[1] for ea in path_actions]


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    search_queue = util.Queue()

    successors = problem.getSuccessors(problem.getStartState())

    for ea in successors:
        search_queue.push(ea)


    find_goal = False

    path_actions = []

    visited_pos = set()
    visited_pos.add(problem.getStartState())

    #开始遍历：如果队列不空，且没有到达目标结点：
    while not search_queue.isEmpty() and find_goal == False:
        choice = search_queue.pop()
        if not problem.isGoalState(choice[0]):
            # 如果该节点没被访问
            if choice[0] not in visited_pos:
                visited_pos.add(choice[0])
                path_actions.append(choice)

            # filter的意思是对sequence中的所有item依次执行 function(item)
            choice_successors = filter(lambda v: v[0] not in visited_pos, problem.getSuccessors(choice[0]))

            #广度优先遍历，把可前进的节点加入队列
            for ea in choice_successors:
                search_queue.push(ea)


        else:
            #走到终点了
            path_actions.append(choice)
            visited_pos.add(choice[0])
            find_goal = True

            #从终点开始往回找，获得路径
            path = []
            temp = list(choice[0])
            path.append(choice[1])

            while temp != list(problem.getStartState()):
                temp = list(choice[0])
                if choice[1] == "North":
                    temp[1] -= 1
                elif choice[1] == "South":
                    temp[1] += 1
                elif choice[1] == "West":
                    temp[0] += 1
                elif choice[1] == "East":
                    temp[0] -= 1

                tp = tuple(temp)

                for ea in path_actions:
                    if tp == ea[0]:
                        choice = ea
                        #把路径加入路径数组
                        path.append(choice[1])
                        break
            path.reverse()

    return [ea for ea in path]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #声明一个优先队列，优先队列利用最小堆的原理实现，优先级低的先pop出
    search_queue = util.PriorityQueue()
    search_list = []
    find_goal = False
    path_actions = []

    # 找到起点周围能到达的点
    successors = problem.getSuccessors(problem.getStartState())
    visited_pos = set()
    # 起点已访问
    visited_pos.add(problem.getStartState())
    # 把起点周围能到的点加入搜索队列
    for ea in successors:
        search_queue.push([ea, 1], 1)
        search_list.append([ea, 1])

    # 如果搜索队列不为空且没到终点
    while not search_queue.isEmpty() and find_goal == False:
        # 弹出目前权值最低的点，以此点为出发继续搜索
        choice = search_queue.pop()
        search_list.remove(choice)
        if not problem.isGoalState(choice[0][0]):
            if choice[0][0] not in visited_pos:
                visited_pos.add(choice[0][0])
                path_actions.append(choice)
            #搜索当前点周围能到的点
            choice_successors = filter(lambda v: v[0] not in visited_pos, problem.getSuccessors(choice[0][0]))

            #更新他们的权值
            for ea in choice_successors:
                newNode = [ea, 0]
                #计算到这个点的权值
                newNode[1] = choice[1] + abs(ea[0][0] - choice[0][0][0]) + abs(ea[0][1] - choice[0][0][1])

                #检查搜索到的点是否已经在搜索列表中存在
                flag = 0
                for i in search_list:
                    if ea == i[0]:
                        flag = 1
                        #如果存在，看看新的路径是否比旧的更优，如果更优则更新路径权值和父节点
                        if newNode[1] < i[1]:
                            #更新父节点
                            father = ""
                            if ea[0][0] == choice[0][0][0] - 1:
                                father = "East"
                            elif ea[0][0] == choice[0][0][0] + 1:
                                father = "West"
                            elif ea[0][1] == choice[0][0][1] - 1:
                                father = "North"
                            elif ea[0][1] == choice[0][0][1] + 1:
                                father = "South"
                        #把搜索队列里原来的这个点弹出去
                        search_queue.update(i, 0)
                        search_queue.pop()
                        search_list.remove(i)
                        #把更新好的这个点放回搜索队列
                        search_queue.push([(ea(0), father, ea(2)), newNode[1]], newNode[1])
                        search_list.append([(ea(0), father, ea(2)), newNode[1]])
                        break
                # 如果搜索到的点不存在在搜索队列中，那么把他加入搜索队列
                if flag == 0:
                    search_queue.push(newNode, newNode[1])
                    search_list.append(newNode)


        else:
            #到终点了
            path_actions.append(choice)
            visited_pos.add(choice[0][0])
            find_goal = True

            path = []
            temp = list(choice[0][0])
            path.append(choice[0][1])

            #从终点开始沿着父节点反向推出起点到终点的路径
            while temp != list(problem.getStartState()):
                temp = list(choice[0][0])
                if choice[0][1] == "North":
                    temp[1] -= 1
                elif choice[0][1] == "South":
                    temp[1] += 1
                elif choice[0][1] == "West":
                    temp[0] += 1
                elif choice[0][1] == "East":
                    temp[0] -= 1

                tp = tuple(temp)

                for ea in path_actions:
                    if tp == ea[0][0]:
                        choice = ea

                        path.append(choice[0][1])
                        break
            path.reverse()

    return [ea for ea in path]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    OpenList = list()
    CloseList = set()
    find_goal = False#是否找到终点
    path_actions = []#保存路径

    #获取起点周围能到达的点
    successors = problem.getSuccessors(problem.getStartState())

    #A*算法核心公式 F = G + H
    for ea in successors:
        #G值代表起点走到当前点需要的最小花费
        G = 0 + abs(ea[0][0] - problem.getStartState()[0]) + abs(ea[0][1] - problem.getStartState()[1])
        #H值代表估算当前点到终点的距离，这里用欧拉距离估算
        H = math.sqrt((ea[0][0] - 1)**2 + (ea[0][1] - 1)**2)
        F = G + H
        # F = H
        temp1 = [ea, G, F]
        #把新探索到的点加入OpenList
        OpenList.append(temp1)

    #把起点加入CloseList，不再理他
    CloseList.add(problem.getStartState())

    while find_goal == False and len(OpenList) != 0:
        F = 99999999999999999999999#先把F值初始化最大
        nextnode = []

        #在OpenList里找一个F值最小的点，移动到这个点，后文的当前点就是这个F值最小的点
        for node in OpenList:
            if node[2] < F:
                F = node[2]
                nextnode = node
        # 把这个点从OpenList中去除
        OpenList.remove(nextnode)
        # 把这个点加入CloseList
        CloseList.add(nextnode[0][0])
        path_actions.append(nextnode[0])

        # 如果这个点不是终点说明我们要继续搜索
        if not problem.isGoalState(nextnode[0][0]):
            #找到所有与这个点相邻可前进并且不在CloseList中的点
            choice_successors = filter(lambda v: v[0] not in CloseList, problem.getSuccessors(nextnode[0][0]))
            #计算所有这些点的F值
            for ea in choice_successors:
                flag = 0
                G = 0
                tempnode = []
                #在这里考虑这样一个问题，如果这个新发现的点其实已经发现过，在OpenList中已经存在，那么当前点到这个已经在OpenList里的点的距离加上当前点G值是否比这个点本来的G值小
                #也就是说当前点到这个已经在OpenList里的点的距离是否更优
                for i in OpenList:
                    if i[0] == ea:
                        flag = 1
                        tempnode = i
                # 如果新发现的点不在OpenList中就按照普通方法计算F值
                if flag == 0:
                    G = nextnode[1] + abs(ea[0][0] - nextnode[0][0][0]) + abs(ea[0][1] - nextnode[0][0][1])
                    H = math.sqrt((ea[0][0] - 1) ** 2 + (ea[0][1] - 1) ** 2)
                    F = G + H
                    # F = H
                    temp2 = [ea, G, F]
                    OpenList.append(temp2)
                # 如果新发现的点其实已经发现过在OpenList中存在
                elif flag == 1:
                    g = nextnode[1] + abs(ea[0][0] - nextnode[0][0][0]) + abs(ea[0][1] - nextnode[0][0][1])
                    #那么看看从当前点到这个新发现的点在走到当前点这条路是否更近
                    if g < tempnode[1]:
                        #如果更近，那么把这个已经存在在OpenList中的点的G值F值和父节点更新
                        father = ""
                        if ea[0][0] == nextnode[0][0][0] - 1:
                            father = "East"
                        elif ea[0][0] == nextnode[0][0][0] + 1:
                            father = "West"
                        elif ea[0][1] == nextnode[0][0][1] - 1:
                            father = "North"
                        elif ea[0][1] == nextnode[0][0][1] + 1:
                            father = "South"
                        H = math.sqrt((ea[0][0] - 1) ** 2 + (ea[0][1] - 1) ** 2)
                        F = g + H
                        # F = H
                        update = [(ea[0], father, ea[2]), g, F]
                        OpenList.remove(tempnode)
                        OpenList.append(update)

        else:#到达终点
            find_goal = True
            # 从终点开始往回找，获得路径
            path = []
            temp = list(nextnode[0][0])
            path.append(nextnode[0][1])
            choice = nextnode[0]

            while temp != list(problem.getStartState()):
                temp = list(choice[0])
                if choice[1] == "North":
                    temp[1] -= 1
                elif choice[1] == "South":
                    temp[1] += 1
                elif choice[1] == "West":
                    temp[0] += 1
                elif choice[1] == "East":
                    temp[0] -= 1

                tp = tuple(temp)

                for ea in path_actions:
                    if tp == ea[0]:
                        choice = ea
                        # 把路径加入路径数组
                        path.append(choice[1])
                        break
            path.reverse()

    return [ea for ea in path]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs

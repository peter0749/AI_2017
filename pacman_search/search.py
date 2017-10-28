# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).

  You do not need to change anything in this class, ever.
  """

  def getStartState(self):
     """
     Returns the start state for the search problem
     """
     util.raiseNotDefined()

  def isGoalState(self, state):
     """
       state: Search state

     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state

     For a given state, this should return a list of triples,
     (successor, action, stepCost), where 'successor' is a
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take

     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()


def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].

  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].

  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:

  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  util.raiseNotDefined()
  """
  "*** YOUR CODE HERE ***"
  from collections import deque
  initState = problem.getStartState() ## coord
  if problem.isGoalState(initState):
      return []
  pa = {initState:None}
  stk = deque() ## not limit the size
  goalState = None
  solution = []
  stk.append(initState) ## current
  while len(stk)>0 and goalState is None:
      currState = stk.pop() ## stack behavior
      nextStates = problem.getSuccessors(currState)
      for node, Dir, _ in nextStates:
          if not node in pa:
              pa[node] = (currState, Dir) #bt
              if problem.isGoalState(node):
                  goalState = node
                  break
              stk.append(node)
  del stk
  if not goalState is None:
    currState = goalState
    while currState in pa and not pa[currState] is None:
        (lastState, Dir) = pa[currState]
        solution.insert(0, Dir)
        currState = lastState
    return solution
  return []

def breadthFirstSearch(problem):
  "Search the shallowest nodes in the search tree first. [p 81]"
  "*** YOUR CODE HERE ***"
  """
  util.raiseNotDefined()
  """
  from collections import deque
  initState = problem.getStartState() ## coord
  if problem.isGoalState(initState):
      return []
  pa = {initState:None}
  queue = deque() ## not limit the size
  goalState = None
  solution = []
  queue.append(initState) ## current
  while len(queue)>0 and goalState is None:
      currState = queue.popleft() ## queue behavior
      nextStates = problem.getSuccessors(currState)
      for node, Dir, _ in nextStates:
          if not node in pa:
              pa[node] = (currState, Dir) #bt
              if problem.isGoalState(node):
                  goalState = node
                  break
              queue.append(node)
  del queue
  if not goalState is None:
    currState = goalState
    while currState in pa and not pa[currState] is None:
        (lastState, Dir) = pa[currState]
        solution.insert(0, Dir)
        currState = lastState
    return solution
  return []

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  from collections import deque
  initState = problem.getStartState() ## coord
  if problem.isGoalState(initState):
      return []
  pa = {initState:None}
  queue = deque() ## not limit the size
  goalState = None
  solution = []
  queue.append(initState) ## current
  while len(queue)>0 and goalState is None:
      currState = queue.popleft() ## queue behavior
      nextStates = problem.getSuccessors(currState)
      for node, Dir, _ in nextStates:
          if not node in pa:
              pa[node] = (currState, Dir) #bt
              if problem.isGoalState(node):
                  goalState = node
                  break
              queue.append(node)
  del queue
  if not goalState is None:
    currState = goalState
    while currState in pa and not pa[currState] is None:
        (lastState, Dir) = pa[currState]
        solution.insert(0, Dir)
        currState = lastState
    return solution
  return []

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  import heapq
  initState = problem.getStartState() ## coord
  if problem.isGoalState(initState):
      return []
  pa = {initState:None}
  fScore = dict()
  gScore = dict()
  queue = [] ## not limit the size
  goalState = None
  solution = []
  gScore[initState]=0 ## real
  fScore[initState]=heuristic(initState,problem) ## estimated
  heapq.heappush(queue, (fScore[initState], initState))
  while len(queue)>0 and goalState is None:
      score, currState = heapq.heappop(queue) ## min heap behavior
      if score>fScore[currState]: continue ## out-dated information
      nextStates = problem.getSuccessors(currState)
      for node, Dir, _ in nextStates:
          if node in pa: continue ## explored new node
          gScore[node] = gScore[currState]+1
          fScore[node] = gScore[node] + heuristic(node,problem)
          pa[node] = (currState, Dir) #bt
          if problem.isGoalState(node):
              goalState = node
              break
          heapq.heappush(queue,(fScore[node], node))
  del queue
  if not goalState is None:
    currState = goalState
    while currState in pa and not pa[currState] is None:
        (lastState, Dir) = pa[currState]
        solution.insert(0, Dir)
        currState = lastState
    return solution
  return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

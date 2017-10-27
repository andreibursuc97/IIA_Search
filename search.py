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

import util
import heapq


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
    print "Start's successors:", problem.getSuccessors(problem.getStartState()) problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    frontier = util.Stack()
    node = Node(problem.getStartState(), None, None, None)
    if problem.isGoalState(node.getPosition()):
        return []
    visited = []

    frontier.push(node)
    visitedNode = []

    actions = []
    while (not frontier.isEmpty()):
        node = frontier.pop()

        if visited.count(node.getPosition()) == 1:
            continue

        visited.insert(0, node.getPosition())
        visitedNode.insert(0, node)


        actions = []
        node2 = visitedNode[-len(visitedNode)]

        while (node2.getAction() != None):
            actions.insert(0, node2.getAction())
            node2 = node2.getParent()

        if (problem.isGoalState(node.getPosition())):
            break;

        for children in problem.getSuccessors(node.getPosition()):

            node2 = Node(children[0], node, children[1], children[2])

            if visited.count(node2.getPosition()) == 0:

                frontier.push(node2)

    return actions
"""
    Frontier = util.Stack()
    Visited = []
    Frontier.push((problem.getStartState(), []))
    Visited.append(problem.getStartState())

    while Frontier.isEmpty() == 0:
        state, actions = Frontier.pop()

        for next in problem.getSuccessors(state):
            n_state = next[0]
            n_direction = next[1]
            if n_state not in Visited:
                if problem.isGoalState(n_state):
                    return actions + [n_direction]
                else:
                    Frontier.push((n_state, actions + [n_direction]))
                    Visited.append(n_state)
 """
"""util.raiseNotDefined()"""

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    frontier = util.Queue()
    node = Node(problem.getStartState(), None, None, None)
    if problem.isGoalState(node.getPosition()):
        return []
    visited = []
    # i=2
    frontier.push(node)
    visitedNode=[]

    actions = []
    while (not frontier.isEmpty()):
        node = frontier.pop()

        if visited.count(node.getPosition()) == 1:
            continue
        visited.insert(0, node.getPosition())
        visitedNode.insert(0,node)

        actions=[]
        node2=visitedNode[-len(visitedNode)]
        while(node2.getAction()!=None):
            actions.insert(0, node2.getAction())
            node2 = node2.getParent()

        if (problem.isGoalState(node.getPosition())):
            break;

        for children in problem.getSuccessors(node.getPosition()):
            node2 = Node(children[0], node, children[1], children[2])
            if visited.count(node2.getPosition()) == 0:
                frontier.push(node2)

    return actions

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """frontier = []
    heapq.heapify(frontier)"""
    frontier=util.PriorityQueue()
    frontier2=[]
    node = Node(problem.getStartState(), None, None, None)
    visited=[]
    frontier.push(node,node.getCost())
    frontier2.insert(0,node.getPosition())
    visitedNode = []
    actions=[]
    #print "frontierprim"
    #print frontier2
    #i=0
    while(not frontier.isEmpty()):
        node=frontier.pop()
        frontier2.remove(node.getPosition())
        visited.insert(0, node.getPosition())
        visitedNode.insert(0, node)
        #print node.getCost()

        if (problem.isGoalState(node.getPosition())):
            actions = []
            node2 = visitedNode[-len(visitedNode)]
            #print node.getPosition()
            while (node2.getAction() != None):
                # print "repeta"
                actions.insert(0, node2.getAction())
                node2 = node2.getParent()
            break;


        # print node[1].getPosition()
        #print problem.getSuccessors(node.getPosition())
        for children in problem.getSuccessors(node.getPosition()):
            # print "for"
            #print children[2]
            node2 = Node(children[0], node, children[1], children[2])
            #print children[0], children[1], children[2]
            if visited.count(node2.getPosition()) == 0 and (frontier2.count(node2.getPosition())==0 or problem.isGoalState(node2.getPosition())) :
                if node.getCost() is not None:
                    node2.setCost(node2.getCost()+node.getCost())
                frontier.push(node2,node2.getCost())
                frontier2.insert(0,node2.getPosition())
               # print "frontier"
                #print frontier2
                #print i
                #i=i+1

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    frontier2 = []
    node = Node(problem.getStartState(), None, None, None)
    visited = []
    frontier.push(node, node.getCost())
    frontier2.insert(0, node.getPosition())
    visitedNode = []
    actions = []

    i=0
    while (not frontier.isEmpty()):
        node = frontier.pop()
        frontier2.remove(node.getPosition())

        visited.insert(0, node.getPosition())
        visitedNode.insert(0, node)

        if (problem.isGoalState(node.getPosition())):
            actions = []
            node2 = visitedNode[-len(visitedNode)]
            # print node.getPosition()
            while (node2.getAction() != None):
                # print "repeta"
                actions.insert(0, node2.getAction())
                node2 = node2.getParent()
            break;


        for children in problem.getSuccessors(node.getPosition()):

            node2 = Node(children[0], node, children[1], children[2])

            if visited.count(node2.getPosition()) == 0 and (frontier2.count(node2.getPosition()) == 0 or problem.isGoalState(node2.getPosition())):
                if node.getCost() is not None:
                    cost=node2.getCost() + node.getCost()+heuristic(node2.getPosition(),problem)
                    node2.setCost(node2.getCost() + node.getCost())

                else:
                    cost = node2.getCost()+ heuristic(node2.getPosition(), problem)

                frontier.push(node2, cost)
                frontier2.insert(0, node2.getPosition())

            """else:
                #print problem.isGoalState(node.getPosition())
                if frontier2.count(node2.getPosition()) == 1 and problem.isGoalState(node2.getPosition()):
                    list2 = []
                    pop = frontier.pop()
                    frontier2.remove(pop.getPosition())
                    while (not frontier.isEmpty() and pop.getPosition() != node2.getPosition()):
                        list2.insert(0, pop)
                        pop = frontier.pop()
                        frontier2.remove(pop.getPosition())
                    if node.getCost() is not None:
                        cost = node2.getCost() + node.getCost() + heuristic(node2.getPosition())
                        node2.setCost(node2.getCost() + node.getCost())
                    else:
                        cost = node2.getCost() + heuristic(node2.getPosition())

                    frontier.push(node2, cost)
                    frontier2.insert(0, node2.getPosition())
                    print "else cost"
                    print node2.getPosition()
                    print cost
                    if problem.isGoalState(node2.getPosition()) or node2.getCost()<pop.getCost():
                        frontier.push(node2, node2.getCost())
                        frontier2.insert(0, node2.getPosition())
                    else:
                        list2.insert(0,pop)
                
                    while (list2):
                        node3 = list2.pop()
                        frontier.push(node3, node3.getCost())
                        frontier2.insert(0, node3.getPosition())
                    print "frontiergoal"
                    print frontier2
                    """
            """
             cost = node2.getCost() + node.getCost()+heuristic(node2.getPosition(),problem)
            else:
                if frontier2.count(node2.getPosition())==1:

                    list2=[]
                    pop=frontier.pop()
                    frontier2.remove(pop.getPosition())
                    while(not frontier.isEmpty() and pop.getPosition()!=node2.getPosition()):
                        list2.insert(0, pop)
                        pop = frontier.pop()
                        frontier2.remove(pop.getPosition())
                    if node.getCost() is not None:
                        cost = node2.getCost() + node.getCost()
                    else:
                        cost = node2.getCost()
                    frontier.push(node2, cost)
                    frontier2.insert(0, node2.getPosition())
                    while(list2):
                        node3=list2.pop()
                        frontier.push(node3,node3.getCost())
                        frontier2.insert(0,node3.getPosition())

            # print visited.count(node2.getPosition())
            # frontier.insert(0, node)
"""
    return actions
    util.raiseNotDefined()


class Node:
    def __init__( self,position,parent, action, cost):

        self.position = position  # the attribute name of the class CustomeNode

        self.cost = cost  # the attribute cost of the class CustomeNode
        self.parent=parent
        self.action=action

    def getPosition(self):
        return self.position

    def getCost(self):
        return self.cost

    def getParent(self):
        return self.parent

    def getAction(self):
        return self.action

    def setAction(self,action):
        self.action=action

    def setCost(self,cost):
        self.cost=cost



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

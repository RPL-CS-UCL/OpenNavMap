#!/usr/bin/env python

import rospy
import networkx as nx
import yaml
import spot
import matplotlib.pyplot as plt

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion
from actionlib_msgs.msg import GoalStatus

class LTLTaskPlanner:
    """
    A class to plan and execute robotic tasks specified in Linear Temporal Logic (LTL).
    It loads a topological map, translates an LTL formula into a Büchi automaton,
    finds an optimal plan by searching on the product automaton, and executes it
    using the ROS navigation stack. It also includes visualization for the graphs.
    """
    def __init__(self, map_file_path):
        """Initializes the LTL Task Planner."""
        try:
            rospy.init_node('ltl_task_planner', anonymous=True)
            rospy.loginfo("LTL Task Planner node initialized.")
        except rospy.exceptions.ROSException:
            rospy.logwarn("Node has already been initialized, continuing.")

        self.map_graph = self._load_map_from_yaml(map_file_path)
        rospy.loginfo(f"Topological map loaded with {self.map_graph.number_of_nodes()} nodes.")

        self.product_graph = None
        self.plan_path = None
        self.initial_buchi_state = None
        self.accepting_buchi_states = set()
        
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if not self.move_base_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Could not connect to move_base action server. Running in planning-only mode.")
            self.move_base_client = None
        else:
            rospy.loginfo("Connected to move_base action server.")

    def _load_map_from_yaml(self, file_path):
        """Loads a topological map from a YAML file into a NetworkX directed graph."""
        g = nx.DiGraph()
        with open(file_path, 'r') as f:
            map_data = yaml.safe_load(f)
        for node_data in map_data.get('nodes', []):
            g.add_node(node_data['id'], label=node_data['label'], pose=node_data.get('pose', {}))
        for edge_data in map_data.get('edges', []):
            g.add_edge(edge_data['from'], edge_data['to'], weight=edge_data.get('cost', 1.0))
        return g

    def plan_from_ltl(self, ltl_formula, initial_robot_location_id):
        """
        Generates a plan from an LTL formula by building a product automaton
        and finding the shortest path to an accepting state.
        """
        rospy.loginfo(f"Planning for LTL formula: {ltl_formula}")
        
        # Step 1: Translate LTL formula to a Büchi Automaton using Spot
        try:
            buchi_aut = spot.translate(ltl_formula, 'BA', 'deterministic')
            rospy.loginfo(f"Generated Büchi automaton with {buchi_aut.num_states()} states.")

            self.initial_buchi_state = buchi_aut.get_init_state_number()
            self.accepting_buchi_states = {s for s in range(buchi_aut.num_states()) if buchi_aut.state_is_accepting(s)}
            rospy.loginfo(f"Büchi Automaton Info: Initial State={self.initial_buchi_state}, Accepting States={self.accepting_buchi_states}")

        except Exception as e:
            rospy.logerr(f"Failed to translate LTL formula: {e}")
            return None

        # Step 2: Construct the Product Automaton
        self.product_graph = nx.DiGraph()
        
        atomic_propositions = {str(ap) for ap in buchi_aut.ap()}
        rospy.loginfo(f"Atomic propositions in formula: {atomic_propositions}")

        for map_node in self.map_graph.nodes():
            for buchi_state in range(buchi_aut.num_states()):
                self.product_graph.add_node((map_node, buchi_state))

        for u, v, edge_data in self.map_graph.edges(data=True):
            proposition_at_v = v
            valuation = {ap: (ap == proposition_at_v) for ap in atomic_propositions}

            for buchi_src in range(buchi_aut.num_states()):
                for edge in buchi_aut.out(buchi_src):
                    # Check if this transition is enabled by evaluating the condition
                    condition_str = spot.bdd_format_formula(buchi_aut.get_dict(), edge.cond)

                    def evaluate_condition(cond_str, val_dict):
                        """Evaluate a BDD condition string against a valuation dictionary."""
                        if cond_str == '1':
                            return True
                        # Split disjuncts
                        disjuncts = cond_str.split(' | ')
                        for disjunct in disjuncts:
                            disjunct = disjunct.strip()
                            if not disjunct:
                                continue
                            # Split conjuncts
                            conjuncts = disjunct.split(' & ')
                            all_true = True
                            for conjunct in conjuncts:
                                conjunct = conjunct.strip()
                                if conjunct == '1':
                                    continue
                                elif conjunct.startswith('!'):
                                    prop = conjunct[1:]
                                    if val_dict.get(prop, False):  # If prop is true, !prop is false
                                        all_true = False
                                        break
                                else:
                                    prop = conjunct
                                    if not val_dict.get(prop, False):  # If prop is false, conjunct is false
                                        all_true = False
                                        break
                            if all_true:
                                return True
                        return False

                    is_enabled = evaluate_condition(condition_str, valuation)

                    if is_enabled:
                        buchi_dst = edge.dst
                        self.product_graph.add_edge(
                            (u, buchi_src),
                            (v, buchi_dst),
                            weight=edge_data['weight']
                        )
                        
        rospy.loginfo(f"Constructed product automaton with {self.product_graph.number_of_nodes()} nodes and {self.product_graph.number_of_edges()} edges.")

        # Step 3: Search for the shortest path to an accepting state
        start_node = (initial_robot_location_id, self.initial_buchi_state)
        print(f"Start node: {start_node}")
        
        if not self.product_graph.has_node(start_node):
            rospy.logerr(f"Start node {start_node} not found in product graph. Is the initial location valid?")
            return None

        shortest_plan = None
        min_cost = float('inf')

        for map_node in self.map_graph.nodes():
            for acc_buchi_state in self.accepting_buchi_states:
                target_node = (map_node, acc_buchi_state)
                if not self.product_graph.has_node(target_node):
                    continue
                try:
                    print(f"Target node: {target_node}")
                    path = nx.shortest_path(self.product_graph, source=start_node, target=target_node, weight='weight')
                    cost = nx.shortest_path_length(self.product_graph, source=start_node, target=target_node, weight='weight')
                    if cost < min_cost:
                        min_cost = cost
                        shortest_plan = path
                except nx.NetworkXNoPath:
                    continue
        
        if not shortest_plan:
            rospy.logwarn("No path to an accepting state found in the product automaton.")
            return None

        self.plan_path = shortest_plan
        final_plan = [map_node for map_node, _ in shortest_plan]
        rospy.loginfo(f"Plan found with {len(final_plan)} steps and cost {min_cost}: {' -> '.join(final_plan)}")
        return final_plan

    def execute_plan(self, plan):
        """Executes a given plan sequence by sending navigation goals."""
        if not self.move_base_client:
            rospy.logerr("Move base client not available. Cannot execute plan.")
            return False

        plan_to_execute = plan[1:] if len(plan) > 1 else []
            
        if not plan_to_execute:
            rospy.loginfo("Plan is empty or only contains the start node. Nothing to execute.")
            return True
        
        rospy.loginfo(f"Executing plan: {' -> '.join(plan_to_execute)}")
        for waypoint_id in plan_to_execute:
            node_data = self.map_graph.nodes[waypoint_id]
            label = node_data.get('label', '')
            
            if 'wait_' in label:
                try:
                    duration = float(label.split('_')[1].replace('s', ''))
                    rospy.loginfo(f"Executing wait action for {duration} seconds.")
                    rospy.sleep(duration)
                except (ValueError, IndexError):
                    rospy.logwarn(f"Could not parse wait duration from label: {label}")
                continue
            
            rospy.loginfo(f"Moving to waypoint: {waypoint_id} ({label})")
            if not self._send_nav_goal(waypoint_id):
                rospy.logerr(f"Failed to reach waypoint {waypoint_id}. Aborting plan.")
                return False
        
        rospy.loginfo("Plan execution completed successfully.")
        return True

    def _send_nav_goal(self, waypoint_id):
        """Sends a navigation goal to move_base and waits for the result."""
        target_pose_data = self.map_graph.nodes[waypoint_id].get('pose')
        if not target_pose_data:
            rospy.logwarn(f"Waypoint {waypoint_id} has no pose data. Skipping.")
            return True

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(
            Point(target_pose_data['x'], target_pose_data['y'], 0.0),
            Quaternion(0, 0, target_pose_data.get('z', 0.0), target_pose_data.get('w', 1.0))
        )

        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        
        state = self.move_base_client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo(f"Successfully reached waypoint {waypoint_id}.")
            return True
        else:
            rospy.logerr(f"Navigation to {waypoint_id} failed with status code: {state}")
            return False

    def visualize_graphs(self):
        """Visualizes the map graph and the product automaton graph."""
        if not self.map_graph:
            print("Map graph is not loaded.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle('LTL Task Planning Visualization', fontsize=16)

        ax1 = axes[0]
        map_pos = nx.spring_layout(self.map_graph, seed=42)
        map_labels = {n: f"{d['label']}\n({n})" for n, d in self.map_graph.nodes(data=True)}
        edge_labels = nx.get_edge_attributes(self.map_graph, 'weight')
        nx.draw(self.map_graph, pos=map_pos, with_labels=False, ax=ax1, node_size=2500, node_color='skyblue', font_color='black')
        nx.draw_networkx_labels(self.map_graph, pos=map_pos, labels=map_labels, font_size=10, ax=ax1)
        nx.draw_networkx_edge_labels(self.map_graph, pos=map_pos, edge_labels=edge_labels, ax=ax1)
        ax1.set_title('Topological Map Graph')

        ax2 = axes[1]
        if self.product_graph and self.product_graph.number_of_nodes() > 0:
            prod_pos = nx.spring_layout(self.product_graph, seed=42, k=0.9, iterations=50)
            
            node_colors = []
            plan_nodes = set(self.plan_path) if self.plan_path else set()
            for node in self.product_graph.nodes():
                if node in plan_nodes:
                    node_colors.append('lightgreen')
                elif node[1] in self.accepting_buchi_states:
                     node_colors.append('gold')
                else:
                    node_colors.append('lightcoral')

            nx.draw(self.product_graph, pos=prod_pos, with_labels=True, ax=ax2, node_size=1200, node_color=node_colors, font_size=8)
            
            if self.plan_path:
                path_edges = list(zip(self.plan_path, self.plan_path[1:]))
                nx.draw_networkx_edges(self.product_graph, pos=prod_pos, edgelist=path_edges, width=3.0, edge_color='green', ax=ax2, arrowsize=20)

            ax2.set_title('Product Automaton Graph (Plan Highlighted)')
        else:
            ax2.text(0.5, 0.5, 'Product graph not generated or is empty.', ha='center', va='center')
            ax2.set_title('Product Automaton Graph')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == '__main__':
    try:
        map_file = "assets/map.yaml" 
        planner = LTLTaskPlanner(map_file)
        
        # NL: "the robot cannot visit all places at once: go to the table, only then go to the chair, only then go to the hallway"
        # ltl_task = "((!n3) U n2) & ((!n5) U n3) & F(n5) & G(! (n2 & n3)) & G(! (n3 & n5)) & G(! (n2 & n5))"
        ltl_task = "((!n3) U n2) & ((!n5) U n3) & F(n5)"

        start_location = 'n1'
        
        plan_sequence = planner.plan_from_ltl(ltl_task, start_location)
        
        if plan_sequence:
            planner.visualize_graphs()
        else:
            rospy.logerr("Failed to generate a valid plan. Cannot visualize.")
            planner.visualize_graphs()

        if plan_sequence and planner.move_base_client:
            rospy.loginfo("Plan generated. Starting execution in 5 seconds...")
            rospy.sleep(0.5)
            planner.execute_plan(plan_sequence)
        elif plan_sequence:
            rospy.loginfo("Plan generated, but move_base is not connected. Skipping execution.")
        else:
            rospy.logerr("Failed to generate a valid plan for the given task.")

    except rospy.ROSInterruptException:
        rospy.loginfo("LTL Task Planner shutting down.")
    except FileNotFoundError:
        rospy.logfatal(f"Error: The map file '{map_file}' was not found. Please check the path.")
    except Exception as e:
        rospy.logfatal(f"An unhandled exception occurred: {e}", exc_info=True)


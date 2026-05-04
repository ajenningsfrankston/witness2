class ObjectDiscoveryAgent:
    """
    An agent that discovers and defines game objects through
    systematic experimentation and hypothesis testing.
    """

    def __init__(self):
        # Knowledge base: stores discovered objects with their properties
        self.object_definitions = {}
        # Hypothesis graph: tracks candidate explanations for observed elements
        self.hypothesis_graph = {}
        # Experimental history: records past actions and their outcomes
        self.experiment_log = []
        # Inventory of identified visual elements not yet defined
        self.unidentified_elements = []

    def discover_objects(self, env, max_experiments=1000):
        """
        Main loop: discover objects through systematic experimentation.
        """
        for experiment_count in range(max_experiments):
            # Get current game state (64x64 grid + metadata)
            obs = env.get_observation()

            # 1. Parse visual elements from the current frame
            visual_elements = self.parse_visual_elements(obs.frame)

            # 2. Identify unfamiliar elements
            new_elements = self.find_new_elements(visual_elements)
            if new_elements:
                self.unidentified_elements.extend(new_elements)

            # 3. Prioritize which element to investigate next
            target_element = self.select_next_element()
            if not target_element:
                # No new elements - use known objects to progress
                action = self.select_goal_directed_action(obs)
                env.step(action)
                continue

            # 4. Generate hypotheses about the element's function
            hypotheses = self.generate_hypotheses(target_element)

            # 5. Test the most promising hypothesis
            test_action = self.design_experiment(target_element, hypotheses[0])

            # 6. Execute and observe results
            pre_state = self.capture_state(obs)
            new_obs = env.step(test_action)
            post_state = self.capture_state(new_obs)

            # 7. Analyze outcome and define the object
            result = self.analyze_experiment(pre_state, post_state, test_action)

            if result['hypothesis_confirmed']:
                # Define the object with its discovered properties
                self.define_object(target_element, result['object_properties'])
                self.hypothesis_graph[target_element.id] = result['confirmed_rule']
            else:
                # Hypothesis failed - record negative evidence
                self.record_negative_evidence(target_element, result)

            # 8. Log experiment for future reference
            self.experiment_log.append({
                'element': target_element,
                'hypothesis': hypotheses[0],
                'action': test_action,
                'result': result
            })

            # 9. Check for level completion
            if self.is_level_complete(new_obs):
                return self.object_definitions

    def parse_visual_elements(self, frame):
        """
        Parse the 64x64 grid into discrete visual elements.
        Returns a list of Element objects with position and color.
        """
        elements = []
        # Use flood fill or connected components to find contiguous regions
        visited = set()
        for y in range(64):
            for x in range(64):
                if (x, y) not in visited and frame[y, x] != 0:  # Not background
                    color = frame[y, x]
                    # Extract the contiguous region
                    region = self.flood_fill(frame, x, y, color)
                    visited.update(region)
                    elements.append(Element(
                        id=f"{color}_{x}_{y}",
                        color=color,
                        bbox=self.get_bounding_box(region),
                        center=self.get_centroid(region),
                        pixels=region
                    ))
        return elements

    def find_new_elements(self, visual_elements):
        """Return elements not yet defined or hypothesized about."""
        new = []
        for elem in visual_elements:
            if elem.id not in self.object_definitions and elem.id not in self.hypothesis_graph:
                new.append(elem)
        return new

    def generate_hypotheses(self, element):
        """Generate testable hypotheses about an element's function."""
        hypotheses = []

        # Hypothesis 1: Element is a COLLECTIBLE (disappears when touched)
        hypotheses.append(Hypothesis(
            element=element,
            predicted_effect="element_disappears",
            test_action=Action.MOVE_TO(element.center),
            success_criteria="element no longer present in next frame"
        ))

        # Hypothesis 2: Element is a OBSTACLE (causes negative feedback)
        hypotheses.append(Hypothesis(
            element=element,
            predicted_effect="game_reset",
            test_action=Action.MOVE_TO(element.center),
            success_criteria="game state resets to level start"
        ))

        # Hypothesis 3: Element is a GOAL (completes the level)
        hypotheses.append(Hypothesis(
            element=element,
            predicted_effect="level_complete",
            test_action=Action.TOUCH(element.center),
            success_criteria="level completion flag becomes True"
        ))

        # Hypothesis 4: Element is a TELEPORTER (changes player position)
        hypotheses.append(Hypothesis(
            element=element,
            predicted_effect="position_change",
            test_action=Action.MOVE_TO(element.center),
            success_criteria="player position != element center after action"
        ))

        # Hypothesis 5: Element is a SWITCH (affects another element)
        hypotheses.append(Hypothesis(
            element=element,
            predicted_effect="other_element_changes",
            test_action=Action.INTERACT_WITH(element.center),
            success_criteria="some other element's state changes"
        ))

        return hypotheses

    def design_experiment(self, element, hypothesis):
        """Create the minimal action needed to test the hypothesis."""
        if hypothesis.test_action.type == "MOVE_TO":
            # Calculate the sequence of actions to reach the element
            current_pos = self.get_player_position()
            path = self.find_path(current_pos, element.center)
            return ActionSequence(path)
        elif hypothesis.test_action.type == "INTERACT_WITH":
            return Action.INTERACT(element.center)
        else:
            return hypothesis.test_action

    def analyze_experiment(self, pre_state, post_state, action):
        """Compare pre- and post-experiment states to determine the result."""
        result = {
            'hypothesis_confirmed': False,
            'object_properties': {},
            'confirmed_rule': None
        }

        # Check if element disappeared (for COLLECTIBLE hypothesis)
        if pre_state.element_present and not post_state.element_present:
            result['hypothesis_confirmed'] = True
            result['object_properties'] = {
                'type': 'collectible',
                'behavior': 'disappears_on_contact',
                'score_effect': post_state.score - pre_state.score
            }
            result['confirmed_rule'] = Rule(
                trigger="player_touches_element",
                effect="element_removed",
                reward_modifier=result['object_properties']['score_effect']
            )

        # Check if level completed (for GOAL hypothesis)
        elif not pre_state.level_complete and post_state.level_complete:
            result['hypothesis_confirmed'] = True
            result['object_properties'] = {
                'type': 'goal',
                'behavior': 'completes_level_on_contact'
            }
            result['confirmed_rule'] = Rule(
                trigger="player_touches_element",
                effect="level_complete",
                reward=1.0
            )

        # Check if game reset (for OBSTACLE hypothesis)
        elif pre_state.level_number > post_state.level_number:
            result['hypothesis_confirmed'] = True
            result['object_properties'] = {
                'type': 'obstacle',
                'behavior': 'resets_level_on_contact',
                'penalty': pre_state.score - post_state.score
            }
            result['confirmed_rule'] = Rule(
                trigger="player_touches_element",
                effect="level_reset",
                penalty=result['object_properties']['penalty']
            )

        # Check if another element changed (for SWITCH hypothesis)
        else:
            changed_elements = self.find_changed_elements(pre_state, post_state)
            if changed_elements and changed_elements != [pre_state.element]:
                result['hypothesis_confirmed'] = True
                result['object_properties'] = {
                    'type': 'switch',
                    'behavior': 'activates_on_contact',
                    'affected_elements': changed_elements
                }
                result['confirmed_rule'] = Rule(
                    trigger="player_touches_element",
                    effect=f"toggle({', '.join(changed_elements)})"
                )

        return result

    def define_object(self, element, properties):
        """Store the discovered object definition in the knowledge base."""
        self.object_definitions[element.id] = {
            'element': element,
            'type': properties['type'],
            'behavior': properties['behavior'],
            'discovery_confidence': 1.0,
            'interaction_history': [properties]
        }

        # Also record the causal rule discovered
        self.add_causal_rule(properties.get('confirmed_rule'))
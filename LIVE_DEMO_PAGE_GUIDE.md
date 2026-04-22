# The Live Demo Page: A Guided Explanation

The Live Demo page is the teaching stage of this project. It is the place where the system stops being a collection of training scripts, saved models, and evaluation summaries, and becomes a visible interactive story. A reader can watch the recommender take actions, observe how the simulated user reacts, and understand how reinforcement learning decisions affect engagement, uncertainty, and final outcomes.

This guide explains the page as if we were studying a small interactive textbook chapter. It moves from the interface that a teammate sees on screen to the environment, policies, and model artifacts that drive that behavior behind the scenes.

## 1. What the Live Demo Page Is For

The Live Demo page is a step-by-step session simulator for the budgeted interactive movie recommender. It is designed to answer a practical question:

"When should the system ask the user a question, and when should it recommend a movie genre immediately?"

The page demonstrates this tradeoff by running one session at a time and showing:

- the chosen scenario controls
- the policy that is making decisions
- the system's evolving belief about user preferences
- the user's simulated reactions
- the changing engagement and uncertainty state
- the final session outcome

In other words, the page is not only a demo. It is also an explanation tool.

## 2. The Main Structure of the Page

The Live Demo page is organized around a simple rhythm:

1. Choose a scenario.
2. Start the session.
3. Advance it step by step or autoplay it.
4. Read the evolving state.
5. Conclude from the final summary.

The page contains five main visual areas:

- `ScenarioControls`: the setup panel
- `Session Timeline`: the story of each step
- `Belief Over Genres`: the recommender's internal belief state
- `Budget and State`: engagement, uncertainty, question usage, and cumulative reward
- `Final Session Summary`: the final verdict of the session

There is also an `Advanced Step Details` section that exposes the raw JSON of the latest step for technical review.

## 3. The Scenario Controls

The top control panel determines what kind of session will be created. It contains:

- `User Profile`
- `Policy`
- `Question Budget`
- `Seed (Optional)`
- action buttons: `Start`, `Next Step`, and `Reset`
- playback buttons: `Play Session` and `Pause Playback`

These controls matter because the Live Demo does not run a generic simulation. It runs a specific scenario created from the chosen inputs.

### User Profile

The `User Profile` dropdown selects the simulated user persona. It does not refer to a real account or a logged-in user. Instead, it chooses one of the predefined behavior types from `src/env/user_simulator.py`.

The four profiles are:

- `action_focused`
- `balanced_viewer`
- `novelty_seeking`
- `question_sensitive`

Each profile has hidden parameters that shape the session:

- base genre preferences
- question tolerance
- repetition sensitivity
- abandonment threshold
- engagement gains or losses from good and bad questions

The dropdown labels shown in the UI are simply title-cased versions of those internal names:

- `Action Focused`
- `Balanced Viewer`
- `Novelty Seeking`
- `Question Sensitive`

The profiles can be understood as follows:

### Action Focused

This profile leans strongly toward action-oriented content. It is not extremely fragile, but it is not endlessly patient either. If the system asks questions that do not help quickly, the profile is willing to disengage.

### Balanced Viewer

This is the most even-tempered profile. Preferences are more evenly distributed, question tolerance is relatively high, repetition sensitivity is low, and abandonment risk is lower than for the more demanding profiles. This often makes it the easiest profile for a policy to handle gracefully.

### Novelty Seeking

This profile tends to prefer less conventional or more exploratory content, such as sci-fi and documentary. It can respond well to good discovery-oriented interactions, but repeated or unhelpful behavior can still produce frustration.

### Question Sensitive

This profile is the least tolerant of questioning. Poor questions hurt engagement more strongly, repetition is more damaging, and abandonment risk rises more quickly. It is a useful profile for testing whether a policy knows when to stop asking and move to recommending.

### Policy

The `Policy` dropdown chooses the decision-maker. Some policies are hand-written baselines, while others are loaded from trained models.

Supported options include:

- `always_recommend`
- `always_ask`
- `ask_once_then_recommend`
- `random_policy`
- `q_learning`
- `dqn`
- `ppo`

The baseline policies are simple rules:

- `always_recommend` immediately recommends and never asks
- `always_ask` keeps cycling through question actions
- `ask_once_then_recommend` asks once, then switches to recommendation
- `random_policy` samples actions without strategy

The RL policies are different:

- `q_learning`
- `dqn`
- `ppo`

These are loaded from saved artifacts under the repo root `models` folder.

### Question Budget

The `Question Budget` sets the maximum number of questions the session is allowed to use before going over budget. It expresses a central idea of this project: questions are not free. Asking can help the system learn, but too much questioning creates friction and hurts the user experience.

### Seed

The `Seed` is used for reproducibility. If the same policy, profile, budget, and seed are used again, the session will tend to replay the same stochastic outcomes.

If the seed field is left blank, the backend falls back to the default config seed, which is currently `42`.

This detail matters a great deal during demos. If the controls are unchanged and the seed remains fixed, the resulting behavior is meant to be reproducible rather than surprising.

## 4. What the Buttons Really Do

The buttons on the Live Demo page are easy to understand visually, but the backend behavior is worth learning carefully.

### Start

`Start` creates a brand-new session using the current control values. This is the button that actually sends the chosen `User Profile`, `Policy`, `Question Budget`, and `Seed` to the backend and asks it to initialize a session.

If a teammate changes dropdowns and wants those new settings to take effect, `Start` must be clicked.

### Next Step

`Next Step` advances the current session by exactly one action. The policy chooses one action, the environment responds, and the timeline grows by one step.

This is the best button for narrated demos because it gives the presenter time to explain what happened at each moment.

### Play Session

`Play Session` does not create a new scenario. It simply auto-advances the existing session at a fixed interval. It is essentially automated clicking of `Next Step`.

This is an important lesson for the team: `Play Session` uses the current `session_id`. It does not read the control panel and start over with newly selected values.

### Reset

`Reset` does not apply newly edited controls either. It resets the existing session using that session's original parameters, including its stored seed and profile. It is a replay of the same scenario, not a fresh scenario built from current dropdown values.

This explains a common source of confusion: if the controls are changed but `Start` is not pressed, `Play Session` and `Reset` can make it seem as if the UI is ignoring the new selections.

## 5. The Session Timeline

The `Session Timeline` is the narrative spine of the page. Each card in the timeline records one system action and one user reaction.

Each step shows:

- step number
- whether the action was `Ask` or `Recommend`
- the exact action name
- the step reward
- the user response
- the current engagement
- the question prompt or recommendation genre

This is the easiest panel to present live because it answers the human question: "What happened next?"

The timeline should be read as the interaction story:

- the policy chose something
- the user reacted
- the state changed
- the session moved closer either to success or failure

## 6. What "User Response" Means

The `User Response` label is built from backend flags and appears as one of three visible outcomes:

- `Accepted`
- `Abandoned`
- `Continued`

These values do not come from a free-form text response. They are derived from environment state:

- a recommendation can be accepted
- the session can be abandoned
- otherwise the interaction simply continues

When the UI shows `Abandoned`, it means the backend set the abandonment flag for that step. It is not just a display preference. It is the formal termination event of the session.

## 7. Engagement, Uncertainty, and Cumulative Reward

The `Budget and State` panel contains three of the most important teaching signals in the whole system.

### Engagement

`Engagement` is a score between `0` and `1` that represents how willing the simulated user still is to continue interacting with the system.

Engagement can go up after a good recommendation or a helpful question, and it can go down after poor questions, rejected recommendations, repetition, or friction.

One can think of engagement as the user's remaining patience and interest.

### Uncertainty

`Uncertainty` is a score between `0` and `1` that represents how unsure the recommender still is about the user's preferences.

Lower uncertainty is better. Helpful questions and successful recommendations tend to reduce uncertainty. Failed or unhelpful interactions can increase it.

One can think of uncertainty as the size of the system's knowledge gap.

### Cumulative Reward

`Cumulative Reward` is the running total of the rewards earned across the session.

This score matters because the RL policies were trained to maximize reward. It combines several effects:

- positive reward for accepted recommendations
- penalties for skipped recommendations
- a cost for asking questions
- penalties for over-budget questioning
- penalties for repetition
- a strong penalty for abandonment
- a bonus based on current engagement

This reward design encodes the project's central tension: ask enough to learn, but not so much that the user is lost.

## 8. The Belief Over Genres Chart

The `Belief Over Genres` chart shows the system's internal estimate of what the user likes among the available genres:

- `action`
- `comedy`
- `drama`
- `scifi`
- `documentary`

Each bar shows the score assigned to one genre in the current belief vector. This chart is not the user's true hidden preference distribution. It is the recommender's best current guess.

At the start of a session, the belief is typically flat because the system does not know enough yet. As informative interactions happen, the belief changes.

When a question is asked, the simulator provides a hinted genre signal, and the environment nudges the belief vector toward that hint. This means the chart usually becomes sharper over time if questioning is informative.

The chart can be interpreted in a simple way:

- a flat chart means the system is still unsure
- one or two taller bars mean the system has formed a stronger belief
- if the chart changes meaningfully after questions, the policy is learning
- if the chart stays flat while engagement falls, the policy is likely failing to learn efficiently

The chart and the scalar `uncertainty` value are related but not identical. The chart is the visible distribution; uncertainty is the compact summary of how unsure the system still is.

## 9. The Final Session Summary

The `Final Session Summary` panel is the closing statement of the demo. It compresses the whole session into a small set of headline numbers:

- `Total Reward`
- `Accepted Recommendations`
- `Skipped Recommendations`
- `Questions Used`
- `Session Length`
- `Abandoned`

This panel is best used after the team has already discussed the timeline and the state panels. It answers the final evaluative question:

"Did this policy perform well overall in this scenario?"

## 10. Why Sessions May Appear to End Around Step 5

It is tempting to believe that the Live Demo has some hidden 5-step limit because sessions often seem to stop around Step 5. In fact, the environment does not define such a rule. The configured `max_steps` value is `12`.

When a session ends early, it is usually because the user abandoned, not because the system was forced to stop at step five.

Why, then, does the same ending appear so often?

The answer is reproducibility.

If no seed is entered in the UI, the backend uses the default seed from the config. With a fixed seed, a fixed profile, and a deterministic inference path, the same scenario is replayed repeatedly. This makes the same abandonment point appear again and again.

So the correct conclusion is not that Step 5 is magical. The correct conclusion is that a particular seeded rollout often reaches abandonment around that point.

## 11. How the Backend Creates Reproducible Sessions

The demo backend creates a session by taking:

- the selected policy
- the selected user profile
- the question budget
- the chosen seed, or the config seed if none is provided

It then resets the environment with those values and stores the session state in memory. From that moment onward, `Next Step`, `Play Session`, and `Reset` operate on the stored session rather than re-reading the current control panel.

This design is useful because it makes sessions stable and explainable. But it also means the presenter must be careful:

- changing dropdowns alone does not change the active session
- `Start` is the boundary between "edited controls" and "real backend scenario"

## 12. Where the Demo Loads Models From

The RL models used by the Live Demo are loaded from the repo-root `models` folder:

- `models/q_learning/<latest-run>`
- `models/dqn/<latest-run>`
- `models/ppo/<latest-run>`

The inference adapter chooses the latest run directory for the selected algorithm unless a model directory is explicitly supplied.

This means the demo is not inventing policy behavior on the fly. It is running the most recent saved model artifacts from the project.

The baseline policies are different. They do not load any model file at all. They are simple hard-coded functions.

## 13. Why the Q-Learning Policy Often Recommends First, Uses Few Questions, and Ends Early

The behavior observed for `q_learning` is not merely a display artifact. It is strongly consistent with the trained model's own evaluation summary.

The saved Q-learning evaluation reports:

- average cumulative reward: about `-2.60`
- acceptance rate: about `0.347`
- abandonment rate: about `0.958`
- average questions asked: about `1.87`
- average session length: about `4.53`

These numbers tell an important story. The tabular Q-learning policy did not learn a strong dialog strategy. Instead, it tends to:

- recommend early
- ask relatively few questions
- lose the user quickly
- end sessions in roughly four to five steps on average

This pattern is likely a consequence of several design realities.

First, the model is a tabular agent with a heavily discretized state representation. It compresses the continuous observation into coarse bins and even reduces the belief vector to its single highest-probability genre. That simplification can make learning brittle.

Second, the training budget for Q-learning is modest at `700` episodes.

Third, the agent is trained across mixed user types rather than a single stable user profile. One table must learn across different preference structures and abandonment sensitivities.

At demo time, inference is greedy with `explore=False`, so the system is not improvising. It is repeating the action pattern that the trained Q-table already prefers in those encountered states.

The result is a model that often behaves as if it wants to recommend early, ask little, and accept early abandonment as a recurring outcome.

## 14. Why the PPO Policy Appears to Always Recommend

The PPO behavior is even more striking. The saved PPO evaluation summary reports:

- average cumulative reward: about `1.40`
- acceptance rate: about `0.384`
- abandonment rate: about `0.983`
- average questions asked: `0.0`
- average session length: about `4.08`

That `average_questions_asked: 0.0` is the crucial signal. It means PPO, as evaluated, never used the `Ask` actions at all.

This is not because PPO was manually instructed to avoid questions. Rather, it is because the training setup encouraged that outcome.

The reward design gives:

- a direct cost for asking a question
- a large positive reward for an accepted recommendation
- a smaller penalty for a skipped recommendation
- engagement-based bonus on top of each step

Questions provide mostly indirect benefit. They may improve belief and reduce uncertainty, but they do not produce a direct positive reward on their own. A recommendation, by contrast, can produce a large immediate gain.

Under such incentives, PPO appears to have learned a shortcut strategy:

"Recommend immediately, hope for an acceptance, and do not pay the guaranteed cost of asking."

This strategy is not especially elegant, and it still suffers from a very high abandonment rate. But it is understandable from the reward structure.

It is also important to remember that PPO was trained for only `6000` timesteps. That is not a large training budget for learning a subtle ask-versus-recommend policy over stochastic users.

At inference time, PPO is also run deterministically, so the Live Demo is showing the learned policy clearly rather than exploring alternative actions.

## 15. Why "Always Recommend" Can Be Learned Even in a System About Asking

At first glance, it may seem contradictory that a project about budgeted interactive recommendation can produce policies that barely ask any questions. In fact, this is one of the most educational outcomes in the whole repository.

The system teaches a classic reinforcement-learning lesson: the policy will optimize the reward it is given, not the story the designers hoped it would tell.

If asking has a guaranteed cost, while recommendation carries a chance of large immediate reward, a policy may learn to skip the exploratory phase entirely. This is especially likely when:

- training time is limited
- state representation is coarse
- user behaviors are mixed
- the reward function does not sufficiently reward information gathering itself

Thus, the Live Demo page becomes more than a product showcase. It becomes a diagnostic lens through which the team can see what the reward function and training setup actually taught the models to value.

## 16. How to Present the Live Demo Page to a Team

When presenting the Live Demo page to teammates, it helps to narrate it in layers.

### Layer 1: What the audience is seeing

Begin with the interface:

"This page simulates one interactive recommendation session. We choose a user profile, select a policy, and then watch how the policy decides whether to ask questions or recommend directly."

### Layer 2: What the policy is trying to balance

Then explain the tradeoff:

"The policy is balancing information gain against user fatigue. Asking can reduce uncertainty, but too much asking can reduce engagement and trigger abandonment."

### Layer 3: How to read the page

Then walk the audience through the panels:

- the timeline shows the sequence of decisions
- the belief chart shows the system's internal estimate of the user's genre preference
- the state panel shows engagement, uncertainty, and reward
- the final summary tells whether the session was successful overall

### Layer 4: What the observed behaviors reveal

Finally, interpret the learned policies honestly:

- Q-learning tends to recommend early, ask little, and often loses the user quickly
- PPO appears to have collapsed into a pure recommendation policy
- these are not UI accidents; they are learned outcomes from the current reward and training setup

This style of presentation turns the page into a research artifact rather than a superficial interface demo.

## 17. The Deeper Lesson of the Live Demo Page

The most valuable lesson in the Live Demo page is not that one policy "wins." The deeper lesson is that the page reveals the relationship between four things:

- the simulator's user psychology
- the reward function
- the training budget and model class
- the visible runtime behavior

When the team sees a policy ask too little, recommend too early, or lose the user quickly, the correct response is not merely to criticize the policy. The more useful response is to ask:

- What incentives did we create?
- What information could the model actually represent?
- How much training opportunity did it have?
- What behavior was truly rewarded?

The page is powerful because it makes those questions visible.

## 18. Final Perspective

The Live Demo page should be understood as both a product demonstration and a teaching instrument. It allows the team to observe the complete chain from policy choice to user response, from hidden model behavior to visible session outcome.

The controls make the scenario concrete. The timeline makes the interaction understandable. The belief chart exposes internal reasoning. The budget and state metrics reveal the health of the session. The summary gives a conclusion. And the model behaviors, especially in Q-learning and PPO, remind us that reinforcement learning faithfully reflects the structure we train into it.

That is why this page matters. It does not simply show what the system does. It teaches why the system behaves that way.

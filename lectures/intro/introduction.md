# Introduction


## What is Reinforcement Learning (RL)?

* Reinforcement learning is what to do to maximize a reward.
* We can give a more "formal" definition.

````{prf:definition} Reinforcement Learning
:label: def-rl

Reinforcement Learning is calculating a function that maps situations to actions.
````

*  We said that we want to maximize a reward, but what is a reward?

```{admonition} Activity
:class: activity
Try to explain what a reward is.
```

*  To maximize a reward the learner can do different actions.
*  If the learner was passive, it could not maximize anything.
*  Usually, the learner start with no prior knowledge about what action it should do.

```{admonition} Activity
:class: activity

What would you do to maximize a reward if you had no idea which action you should do?
```

## A Simple Example: Learning to Navigate

Let's start with a simple scenario everyone can relate to:

**Imagine you're in a new city trying to find the best coffee shop.**

* **Your goal**: Find great coffee (maximize reward)
* **Your actions**: Choose which direction to walk, which shops to try
* **Your feedback**: Coffee quality (immediate), but also learning about the neighborhood (delayed)
* **The challenge**: Balance trying new places vs. returning to known good ones

This captures the essence of reinforcement learning:
1. You have a goal (good coffee)
2. You take actions (choose directions/shops)
3. You get feedback (coffee quality)
4. You learn and improve your strategy over time
5. You must balance exploration (new places) vs exploitation (known good places)

```{admonition} Activity
:class: activity
In the coffee shop example:
1. What would happen if you only went to the first decent shop you found?
2. What would happen if you tried a completely new shop every single day?
3. What's a good strategy for the long term?
```

### Try It Yourself: Coffee Shop Navigator

Let's make this concrete! Below is a simple simulation where you can experience the exploration-exploitation trade-off firsthand.

```{raw} html
<div style="border: 2px solid #4CAF50; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: #f9f9f9;">
    <h4>🏙️ Welcome to Coffee Street!</h4>
    <p>You're new in town and looking for great coffee. Each shop has a hidden quality rating (1-10). Your goal: find the best coffee while minimizing bad experiences.</p>

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 20px 0;">
        <button onclick="visitShop(1)" id="shop1" style="padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer;">
            ☕ Café Alpha<br>
            <span id="visits1">Visits: 0</span><br>
            <span id="avg1">Avg: ?</span>
        </button>
        <button onclick="visitShop(2)" id="shop2" style="padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer;">
            ☕ Bean House<br>
            <span id="visits2">Visits: 0</span><br>
            <span id="avg2">Avg: ?</span>
        </button>
        <button onclick="visitShop(3)" id="shop3" style="padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer;">
            ☕ Coffee Corner<br>
            <span id="visits3">Visits: 0</span><br>
            <span id="avg3">Avg: ?</span>
        </button>
        <button onclick="visitShop(4)" id="shop4" style="padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer;">
            ☕ Daily Drip<br>
            <span id="visits4">Visits: 0</span><br>
            <span id="avg4">Avg: ?</span>
        </button>
        <button onclick="visitShop(5)" id="shop5" style="padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer;">
            ☕ Espresso Elite<br>
            <span id="visits5">Visits: 0</span><br>
            <span id="avg5">Avg: ?</span>
        </button>
        <button onclick="visitShop(6)" id="shop6" style="padding: 15px; font-size: 14px; border: 2px solid #ddd; border-radius: 5px; background: #fff; cursor: pointer;">
            ☕ Fresh Grounds<br>
            <span id="visits6">Visits: 0</span><br>
            <span id="avg6">Avg: ?</span>
        </button>
    </div>

    <div style="margin: 20px 0; padding: 15px; background: #e8f5e8; border-radius: 5px;">
        <strong>📊 Your Journey:</strong><br>
        <div id="results" style="margin-top: 10px;">Click a coffee shop to start your exploration!</div>
        <div style="margin-top: 10px;">
            <strong>Total Visits:</strong> <span id="totalVisits">0</span> |
            <strong>Overall Satisfaction:</strong> <span id="avgSatisfaction">-</span>
        </div>
    </div>

    <div style="margin: 15px 0;">
        <button onclick="resetSimulation()" style="padding: 10px 20px; background: #ff6b6b; color: white; border: none; border-radius: 5px; cursor: pointer;">
            🔄 Reset & Try Different Strategy
        </button>
        <button onclick="revealQualities()" style="padding: 10px 20px; background: #4ECDC4; color: white; border: none; border-radius: 5px; cursor: pointer; margin-left: 10px;">
            🎯 Reveal True Qualities
        </button>
    </div>
</div>

<script>
// Coffee shop simulation
let shopQualities = [7.2, 4.1, 8.7, 5.9, 9.1, 6.3]; // Hidden true qualities
let visits = [0, 0, 0, 0, 0, 0];
let totalRatings = [0, 0, 0, 0, 0, 0];
let visitHistory = [];
let totalVisits = 0;

function visitShop(shopIndex) {
    const quality = shopQualities[shopIndex - 1];
    // Add some noise to make it realistic
    const experience = Math.max(1, Math.min(10, quality + (Math.random() - 0.5) * 2));
    const rating = Math.round(experience * 10) / 10;

    visits[shopIndex - 1]++;
    totalRatings[shopIndex - 1] += rating;
    totalVisits++;

    const avgRating = (totalRatings[shopIndex - 1] / visits[shopIndex - 1]).toFixed(1);

    // Update shop display
    document.getElementById(`visits${shopIndex}`).textContent = `Visits: ${visits[shopIndex - 1]}`;
    document.getElementById(`avg${shopIndex}`).textContent = `Avg: ${avgRating}`;

    // Color code based on average (if visited more than once)
    const shopButton = document.getElementById(`shop${shopIndex}`);
    if (visits[shopIndex - 1] > 1) {
        if (avgRating >= 8) shopButton.style.background = '#c8e6c9';
        else if (avgRating >= 6) shopButton.style.background = '#fff3e0';
        else shopButton.style.background = '#ffcdd2';
    }

    // Add to history
    visitHistory.push({shop: shopIndex, rating: rating, isExploration: visits[shopIndex - 1] === 1});

    // Update results
    updateResults(shopIndex, rating, visits[shopIndex - 1] === 1);
    updateStats();
}

function updateResults(shopIndex, rating, isExploration) {
    const shopNames = ['Café Alpha', 'Bean House', 'Coffee Corner', 'Daily Drip', 'Espresso Elite', 'Fresh Grounds'];
    const action = isExploration ? '🔍 EXPLORED' : '✅ EXPLOITED';
    const color = isExploration ? '#FF9800' : '#4CAF50';

    const resultDiv = document.getElementById('results');
    const newResult = `<div style="margin: 5px 0; padding: 8px; background: ${color}20; border-left: 4px solid ${color}; border-radius: 3px;">
        <strong>${action}:</strong> ${shopNames[shopIndex - 1]} → Rating: ${rating}/10
    </div>`;

    if (resultDiv.innerHTML === 'Click a coffee shop to start your exploration!') {
        resultDiv.innerHTML = newResult;
    } else {
        // Limit to last 3 visits
        const currentResults = resultDiv.innerHTML.split('<div style="margin: 5px 0;').slice(0, 2);
        const limitedResults = currentResults.length > 0 ? '<div style="margin: 5px 0;' + currentResults.join('<div style="margin: 5px 0;') : '';
        resultDiv.innerHTML = newResult + limitedResults;
    }
}

function updateStats() {
    document.getElementById('totalVisits').textContent = totalVisits;

    if (totalVisits > 0) {
        const totalSatisfaction = visitHistory.reduce((sum, visit) => sum + visit.rating, 0);
        const avgSatisfaction = (totalSatisfaction / totalVisits).toFixed(1);
        document.getElementById('avgSatisfaction').textContent = `${avgSatisfaction}/10`;
    }
}

function resetSimulation() {
    visits = [0, 0, 0, 0, 0, 0];
    totalRatings = [0, 0, 0, 0, 0, 0];
    visitHistory = [];
    totalVisits = 0;

    // Reset display
    for (let i = 1; i <= 6; i++) {
        document.getElementById(`visits${i}`).textContent = 'Visits: 0';
        document.getElementById(`avg${i}`).textContent = 'Avg: ?';
        document.getElementById(`shop${i}`).style.background = '#fff';
    }

    document.getElementById('results').innerHTML = 'Click a coffee shop to start your exploration!';
    document.getElementById('totalVisits').textContent = '0';
    document.getElementById('avgSatisfaction').textContent = '-';
}

function revealQualities() {
    const shopNames = ['Café Alpha', 'Bean House', 'Coffee Corner', 'Daily Drip', 'Espresso Elite', 'Fresh Grounds'];
    let revelation = '<strong>True Coffee Quality Revealed:</strong><br>';

    shopQualities.forEach((quality, index) => {
        const qualityDesc = quality >= 8 ? '⭐ Excellent' : quality >= 6 ? '👍 Good' : '😐 Mediocre';
        revelation += `${shopNames[index]}: ${quality}/10 ${qualityDesc}<br>`;
    });

    revelation += '<br><em>Now you can see which shops you should have focused on!</em>';

    document.getElementById('results').innerHTML = revelation;
}
</script>
```

**Try Different Strategies:**
1. **Pure Exploration**: Try each shop once, then decide
2. **Pure Exploitation**: Stick with the first decent shop you find
3. **Balanced**: Mix trying new places with returning to good ones

**Reflection Questions:**
- Which strategy gave you the highest overall satisfaction?
- How many visits did it take to identify the best shop?
- What happened when you stuck with the first "okay" shop you found?

Whether you stuck with your first decent coffee shop or kept exploring new ones, you just experienced the core challenge of reinforcement learning.

## Key Concepts - The Basics

Reinforcement learning has two main characteristics:
* **Trial-and-error search**: Learning by trying things and seeing what works
* **Delayed rewards**: Actions now may have consequences much later

**Why these matter:**
- **Trial-and-error** means no teacher provides "correct" answers - you learn by experience
- **Delayed rewards** means you must connect actions to outcomes that happen later

We'll explore these concepts in detail later, but first let's see the bigger picture of this learning approach.

```{warning}

Reinforcement learning is a name that regroups different concepts:
* It's a type of problem.
* It's also a class of solution methods.
* And it's the field that study the two previous points as well.

You need to understand the distinction.
```

Now that you understand these basic characteristics, let's zoom out to see the complete picture.

## What is the Reinforcement learning problem? (Simplified)

* The reinforcement learning problem is an idea coming from dynamical system theory.
* And more specificaly from the Markov Decision Processes.
* The basic ideas are:
  * A learning agent must *sense* the state of the environment.
  * The agent must be able to take *actions* that affect the state.
  * It must have a *goal* or goals relating to the state of the environment.

```{image} ./img/rl.drawio.png
    :align: center
```

This simple agent-environment loop might look basic, but it powers some of the most impressive AI breakthroughs.

## Why Reinforcement Learning Matters

Now that you understand the basics, let's see why RL has become so important:

### Real-World Success Stories

**Game Playing**: AlphaGo defeated world champions, mastering strategy through self-play

**Autonomous Systems**: Self-driving cars make split-second decisions in complex environments

**Finance & Trading**: Systems optimize investment decisions over time under uncertainty

### What Makes These Problems Special?

Traditional AI approaches fail because these problems require:
1. **Learning without a teacher** - no dataset of "perfect" decisions exists
2. **Handling delayed consequences** - actions now affect outcomes much later
3. **Adapting to change** - environment responds to your actions
4. **Balancing exploration vs exploitation** - try new things vs use current knowledge

You might wonder: why couldn't traditional machine learning solve these problems? Understanding what RL is requires understanding what it's not.

## What Reinforcement learning is not?

Understanding RL is easier when we compare it to other learning paradigms:

| Learning Type | Data Required | Feedback Type | Goal | Example | When It Fails |
|---------------|---------------|---------------|------|---------|---------------|
| **Supervised** | Labeled examples | Immediate correct answers | Predict/classify | Email spam detection | No "correct" action dataset available |
| **Unsupervised** | Unlabeled data | No feedback | Find patterns | Customer segmentation | No clear objective function |
| **Reinforcement** | Environment interaction | Delayed rewards | Maximize cumulative reward | Game playing | Need real-time interaction |

### Key Differences Explained

**Why supervised learning fails for RL problems:**
- **No perfect dataset**: There's no collection of "correct" actions for every situation
- **Context dependency**: The best action depends on long-term consequences, not just current state
- **Interactive nature**: The environment changes based on your actions

**Why unsupervised learning isn't enough:**
- **No objective**: Finding patterns doesn't tell you which actions are good
- **No feedback**: You can't improve without knowing if you're doing well

```{admonition} Activity
:class: activity

Looking at the table above:
1. Why can't you use supervised learning to learn chess strategy?
2. What would unsupervised learning find in a chess game, and why isn't that sufficient?
3. Give an example of a problem where you'd need each type of learning.
```

## The challenges of reinforcement learning.

Reinforcement learning faces unique challenges that make it different from other machine learning approaches:

### 1. The Exploration-Exploitation Trade-off

This is the fundamental challenge in RL:

| Strategy | Description | Pros | Cons | Example |
|----------|-------------|------|------|---------|
| **Exploitation** | Use current knowledge to get reward | Immediate gains | Miss better options | Always go to your favorite restaurant |
| **Exploration** | Try new actions to learn | Discover better options | Short-term costs | Try a new restaurant (might be bad) |
| **Balance** | Mix both strategies | Long-term optimal | Complex to implement | Sometimes try new places, sometimes stick to favorites |

```
Agent Decision Process:

1. Agent faces decision
   ├── Action A: Known good reward (reliable)
   └── Action B: Unknown reward (risky)

2. Two strategies:
   ├── EXPLOIT: Choose A → Get expected reward (but miss learning)
   └── EXPLORE: Try B → Learn about B (might find better option)

3. The dilemma:
   • Pure exploitation = stuck with current best
   • Pure exploration = never use what you learn
   • Need balance for optimal long-term performance
```

**Key insight**: You can't do only one strategy - pure exploitation gets stuck in local optima, pure exploration never uses what you learn.

### 2. The Whole Problem Challenge

* **Complete system**: RL considers the entire problem from start to finish
* **Goal-seeking agent**: Must actively pursue objectives, not just respond to inputs
* **Uncertainty handling**: Must operate effectively despite incomplete information about the environment

```{admonition} Activity
:class: activity
Think of a time when you had to balance exploration vs exploitation in real life. How did you decide when to try something new vs stick with what you know?
```

Understanding these challenges prepares us to examine what makes RL systems work. Every RL agent relies on the same core building blocks.

# Elements of reinforcement learning

Every RL system consists of four key components that work together:

```{admonition} Activity
:class: activity

What are the two elements we talked about that compose reinforcement learning?
```

## The Four Core Elements

| Element | Purpose | Think of it as... | Required? |
|---------|---------|-------------------|-----------|
| **Policy** | Decision maker | The brain that chooses actions |  Essential |
| **Reward Function** | Goal definition | The scoring system | Essential |
| **Value Function** | Long-term predictor | The strategic advisor |  Essential |
| **Model** | Environment simulator | The crystal ball | Optional |

### 1. Policy - The Decision Maker

````{prf:definition} Policy
:label: policy_basic

A policy is a function that maps each state to an action.
````

````{dropdown} **Click to explore Policy details**
**What it does:**
- Defines the behavior of an agent at any given time
- Core of the RL agent - determines all actions
- Can be deterministic (same action every time) or stochastic (probabilistic)

```{image} ./img/policy.drawio.png
:align: center
```
````
**Examples:**
- **Chess**: "If opponent threatens my queen, move it to safety"
- **Trading**: "If price drops 5%, sell 20% of holdings"
- **Navigation**: "If obstacle ahead, turn left"


### 2. Reward Function - The Goal Definition

````{prf:definition} Reward
:label: reward

A reward is a value returned by the environment at a time step $t$.
````

**What it does:**
- Defines the goal of the RL problem
- Provides immediate feedback for actions
- Agent's objective: maximize total reward over time

**Examples:**
- **Game**: +10 for winning, -1 for losing, 0 for draw
- **Robot**: +1 for forward movement, -10 for collision
- **Trading**: +profit for good trades, -loss for bad ones

### 3. Value Function - The Strategic Advisor

````{prf:definition} Value Function
:label: value_function_basic

A value function is a function returning for each state the total expected reward starting from this state.
````

**What it does:**
- Estimates long-term expected reward from each state
- Helps agent think strategically, not just about immediate rewards
- Much harder to determine than immediate rewards

**Key insight**: We seek actions that lead to states with higher **value**, not higher immediate **reward**.

```{admonition} Activity
:class: activity

**Reward vs Value - Which would you choose?**
* State A: immediate reward = 100, long-term value = 3
* State B: immediate reward = 1, long-term value = 5

Why might State B be better despite lower immediate reward?
```

**Examples:**
- **Chess**: Sacrificing a piece (negative reward) to gain better position (higher value)
- **Investment**: Spending money on education (cost now) for better career (future value)
- **Medicine**: Taking bitter medicine (negative reward) for health (long-term value)

### 4. Model - The Crystal Ball (Optional)

````{prf:definition} Model
:label: model

The model of the environment is the representation of the dynamic of the problem.
````

```{dropdown} **Click to explore Model details**
**What it does:**
- Predicts what happens next: given current state and action, what's the next state and reward?
- Allows agent to plan ahead and simulate different strategies
- Not always available or practical to learn

**Two RL Approaches:**

| Type | Has Model? | Characteristics | Examples |
|------|------------|-----------------|----------|
| **Model-based** | ✅ Yes | Can plan ahead, simulate scenarios | Chess engines, route planning |
| **Model-free** | ❌ No | Learn directly from experience | Most game AI, trial-and-error learning |

**Examples:**
- **Chess**: Model knows all game rules and can simulate moves
- **Stock Trading**: Model predicts market reactions to events
- **Robot Navigation**: Model predicts where robot will be after moving
```

These four elements work together like a team - each has a specific role, but their real power comes from how they interact in the learning process.

## Summary

You now understand the core building blocks of reinforcement learning! From our coffee shop example to these fundamental elements, you've seen how RL systems learn through experience.

## Current Limitations & Assumptions

RL is powerful but has important limitations to keep in mind:

### State Representation Challenge
| Challenge | Description | Impact | Example |
|-----------|-------------|--------|---------|
| **State Design** | How to represent the current situation | Huge impact on learning speed | Raw pixels vs processed features |
| **State Space Size** | Too many possible states | Learning becomes very slow | Chess: 10^43 possible positions |
| **Partial Observability** | Can't see everything relevant | Must work with incomplete info | Poker: can't see opponents' cards |

**Key Points:**
- **State is everything**: The quality of state representation determines learning success
- **Not magic**: RL assumes you can define reasonable states, actions, and rewards
- **Computational limits**: Large state spaces require advanced techniques
- **Design matters**: How you frame the problem affects what the agent can learn

```{admonition} Activity
:class: activity
Think about teaching someone to drive:
1. What information (state) do they need to make good decisions?
2. What if they could only see through a small window - how would this affect learning?
3. How would you define "good driving" as rewards?
```

---

## What's Next?

You now have a solid foundation in reinforcement learning concepts! Here's what we've covered:

**Core RL Concepts**: Trial-and-error learning with delayed rewards

**Key Elements**: Policy, rewards, values, and models

**Main Challenge**: Balancing exploration vs exploitation

**Why RL Matters**: Real-world applications where traditional AI fails

**Up Next: Multi-Armed Bandits**

Remember how you had to decide which coffee shops to try? We'll formalize this exact problem as "multi-armed bandits" - the perfect stepping stone from our simple example to more complex RL scenarios.

```{admonition} For Advanced Readers
:class: tip
If you want to dive deeper into theoretical foundations, detailed delayed rewards analysis, and advanced case studies like AlphaGo, check out the **Advanced RL Concepts Appendix** after completing the main course content.
```

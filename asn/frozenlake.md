# Frozen Lake

```{important}
Due date: TBD
```

## Context

In this assignment, you will implement TD(0) and MC apply to the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) problem, and analyze the results by comparing their convergence and policy performance.

---

## Assignment

### Part 1: Implement Temporal Difference (TD) Learning

- **Task:** Implement TD(0) and MC algorithms.
- **Environment Setup:**
  - Frozen Lake: Two versions - $4\times 4$ and $8\times 8$.
  - Use a discount factor ( $\gamma = 0.9$ ).
- **Deliverables for Part 1:**
  - Both algorithms TD(0) and MC.
  - A plot showing the value function convergence over episodes.
  - Analysis of how the different parameters affect the convergence:
    - learning rate ($\alpha$).
    - etc.

### Part 2: Comparison and Analysis
- **Task:** Compare the performance and behavior of both methods (TD(0) and MC).
- **Required Analysis:**
  - Discuss which method converges faster and why.
  - Analyze under what conditions one method outperforms the other.
  - Provide a table comparing key metrics, such as:
    - Number of episodes for convergence.
    - Sensitivity to hyperparameters (e.g., learning rate, episode length).
    - Stability of results.
- **Deliverables:**
  - Report with comparisons and insights.
  - Plots showing side-by-side performance (e.g., convergence speed, value estimates).
  - Explanation of potential trade-offs between both methods.

---

## Evaluation Criteria
- **Correctness of Implementations:** 40%
  - TD(0) and MC methods are implemented correctly and produce expected results.
- **Analysis and Insights:** 30%
  - Depth of analysis in comparing the methods.
  - Clear presentation of convergence behaviors and performance metrics.
- **Code Quality:** 20%
  - Well-commented and organized code.
  - Proper use of libraries and good programming practices.
- **Presentation:** 10%
  - Clear plots, tables, and visualizations.
  - Well-written report with proper formatting and insightful observations.

---

## Submission Guidelines

- Code files (.py) uploaded to Moodle.
- Report in PDF format with relevant plots and analysis.

---

## Resources
- Sutton & Barto, *"Reinforcement Learning: An Introduction"*
- Documentation for environments like OpenAI Gym (optional but useful).

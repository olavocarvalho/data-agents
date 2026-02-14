# Data Agents

A curated collection of skills, agents, and commands for AI coding assistants. These assets extend agent capabilities for data science, experimentation, and analytics workflows.

Skills follow the [Agent Skills](https://agentskills.io/) format. Specs available at <https://agentskills.io/specification>

## 📚 Available Skills (17)

### 🧪 Experimentation & Testing

#### ab-test
**Use when:**
- Designing A/B tests or online controlled experiments
- Calculating sample sizes and power analysis
- Analyzing test results with statistical rigor
- Making decisions about experiment readiness or validity
- Interpreting statistical significance and practical significance

**Categories:** Experimentation, Statistics, A/B Testing

---

#### scientific-critical-thinking
**Use when:**
- Evaluating research claims or experimental results
- Identifying logical fallacies or biases in analysis
- Applying the scientific method to data problems
- Peer-reviewing analysis or experiment designs
- Ensuring rigorous, objective reasoning

**Categories:** Scientific Method, Critical Thinking, Research

---

### 📊 Data Analysis & Exploration

#### data-exploration
**Use when:**
- Starting analysis on a new dataset
- Performing exploratory data analysis (EDA)
- Understanding data distributions and relationships
- Identifying data quality issues or anomalies
- Creating initial visualizations and summaries

**Categories:** EDA, Data Analysis, Data Science

---

#### data-validation
**Use when:**
- Validating data quality and integrity
- Checking data schema and types
- Identifying missing values, outliers, or inconsistencies
- Implementing data quality checks in pipelines
- Creating data validation reports

**Categories:** Data Quality, Validation, Data Engineering

---

#### statistical-analysis
**Use when:**
- Performing hypothesis testing (t-tests, ANOVA, chi-square)
- Calculating confidence intervals and effect sizes
- Generating APA-style statistical reports
- Choosing appropriate statistical tests for your data
- Interpreting p-values and statistical significance

**Categories:** Statistics, Hypothesis Testing, Inference

---

#### statistical-analysis-basic
**Use when:**
- Performing basic statistical tests (t-test, proportion test)
- Computing descriptive statistics (mean, median, variance)
- Creating simple statistical summaries
- Learning fundamental statistical concepts
- Quick statistical validation without complexity

**Categories:** Statistics, Basics, Descriptive Stats

---

### 📈 Visualization

#### plotly
**Use when:**
- Creating interactive visualizations
- Building dashboards with hover interactions
- Making publication-quality figures
- Exporting plots to HTML
- Working with time series, scatter plots, or complex charts

**Categories:** Visualization, Interactive, Plotly

---

#### seaborn
**Use when:**
- Creating statistical visualizations
- Making publication-quality static plots
- Using matplotlib-based plotting
- Creating categorical plots, distributions, or heatmaps
- Applying statistical transformations in plots

**Categories:** Visualization, Statistics, Seaborn

---

#### data-vizualization
**Use when:**
- General data visualization best practices
- Choosing the right chart type for your data
- Designing clear, informative visualizations
- Creating accessible and colorblind-friendly plots
- Following data visualization principles

**Categories:** Visualization, Best Practices, Design

---

#### mermaid-diagrams
**Use when:**
- Creating flowcharts, sequence diagrams, or state machines
- Documenting processes or workflows
- Visualizing system architecture
- Generating diagrams from text descriptions
- Embedding diagrams in markdown documentation

**Categories:** Diagrams, Documentation, Mermaid

---

### 🗣️ Communication & Writing

#### data-storytelling
**Use when:**
- Presenting data insights to stakeholders
- Crafting narratives from analysis results
- Building compelling data-driven arguments
- Structuring analysis reports or presentations
- Translating technical findings to business context

**Categories:** Communication, Storytelling, Presentation

---

#### writing-clearly-and-concisely
**Use when:**
- Writing technical documentation
- Drafting emails, reports, or presentations
- Simplifying complex explanations
- Improving clarity and reducing wordiness
- Following technical writing best practices

**Categories:** Writing, Communication, Documentation

---

### 🤖 Machine Learning

#### tabular-ml-modeling
**Use when:**
- Building machine learning models on tabular data
- Using gradient boosting (XGBoost, LightGBM, CatBoost)
- Performing feature engineering and selection
- Tuning hyperparameters
- Evaluating model performance

**Categories:** Machine Learning, Gradient Boosting, Modeling

---

#### shap
**Use when:**
- Explaining machine learning model predictions
- Computing feature importance with SHAP values
- Generating SHAP plots (waterfall, beeswarm, bar, scatter, force, heatmap)
- Debugging or validating model behavior
- Analyzing model bias or fairness
- Implementing explainable AI in production

**Categories:** Machine Learning, Explainability, Model Interpretation, XAI

---

#### fklearn
**Use when:**
- Using Nubank's fklearn functional ML library
- Building functional ML pipelines
- Working with immutable transformations
- Composing feature engineering steps
- Following functional programming patterns in ML

**Categories:** Machine Learning, fklearn, Functional Programming

---

### ⚙️ Platform & Tools

#### databricks-sql
**Use when:**
- Working with arrays, maps, nested data, or JSON/variant types
- Using window functions with QUALIFY
- Querying historical data with Delta Lake time travel
- Using higher-order functions (TRANSFORM, FILTER)
- Analyzing query execution plans with EXPLAIN

**Categories:** SQL, Databricks, Spark

---

#### scala-engineer
**Use when:**
- Writing Scala code following best practices
- Building functional Scala applications
- Working with Scala collections and type systems
- Implementing type-safe designs
- Reviewing or refactoring Scala code

**Categories:** Scala, Programming, Software Engineering

---

## 🤖 Available Agents (1)

### ab-test-analyst
A specialized agent for A/B testing and online controlled experiments. Expert in statistical test design, power analysis, sample size calculation, and result interpretation using both frequentist and Bayesian methods.

**Use when:**
- Designing new experiments
- Analyzing test results
- Troubleshooting unexpected experiment outcomes
- Making go/no-go decisions on tests
- Calculating test duration or sample requirements

---

## 🚀 Usage

### For Claude Desktop / Code
1. Place skills in `.claude/skills/`
2. Place agents in `.claude/agents/`
3. Skills and agents are automatically available when relevant tasks are detected

---

## 📁 Structure

Each skill contains:
- `SKILL.md` - Instructions and context for the agent (required)
- `scripts/` - Helper scripts for automation (optional)
- `references/` - Supporting documentation (optional)
- `assets/` - Templates or resources (optional)

Each agent is a single markdown file with:
- Frontmatter metadata (name, description, model, tools)
- Core expertise and responsibilities
- When to invoke the agent
- Workflow patterns and best practices

---

## 📄 License

MIT

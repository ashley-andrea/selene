### Recommended Tech Stack (Online & Free-Oriented)

The system is designed to remain lightweight, cloud-native, and deployable using free tiers or open infrastructure.

**Model APIs**

* Hosted on Red Hat OpenShift
* Cluster Model API
* Simulator Model API

Note:

* The Safe Gate Engine is NOT deployed as a microservice.
* It is implemented as a deterministic rule module inside the backend to avoid unnecessary infrastructure complexity and latency.

**Backend (Agent Orchestration Layer)**

* Framework: FastAPI (Python)
* Role: State management, agent loop execution, tool orchestration
* Deployment Options (Free-Friendly):
  * Render

**Frontend (User Interface)**

* Framework: Next.js (React)
* Deployment: Vercel (recommended)

**Data Layer (Static Structured Data)**

* CSV Files (Versioned Datasets)

Implementation Strategy:

* CSV files loaded directly by FastAPI at startup
* Managed as Pandas DataFrames in memory
* Accessed via internal data access utilities (pill_database.query())

**Graph & Visualization Layer**

* Handled entirely in Frontend
* Library Options:

  * Recharts
  * Chart.js
  * Visx

Graphs display:

* Risk evolution over time
* Comparative pill trajectories

**Infrastructure Philosophy**

* Stateless backend services
* Models isolated as APIs
* Agent logic centralized
* Deterministic safety layer outside LLM visibility

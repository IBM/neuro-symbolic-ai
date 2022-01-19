# IBM Neuro-Symbolic AI Toolkit (NSTK)

URL: [https://ibm.biz/nstoolkit](https://ibm.biz/nstoolkit)

The Neuro-Symbolic AI (NS) initiative at IBM aims to conceive a fundamental new methodology for AI, to address the gaps remaining between today's state-of-the-art and the full goals of AI, including AGI.  In particular it is aimed at augmenting (and retaining) the strengths of statistical AI (machine learning) with the complementary capabilities of symbolic AI (knowledge and reasoning). It is aimed at a construction of new paradigms rather than superficial synthesis of existing paradigms, and revolution rather than evolution. The primary goals of NS are to demonstrate the capability to:

1. Solve much harder problems
1. Learn with dramatically less data, ultimately for a large number of tasks rather than one narrow task
1. Provide inherently understandable and controllable decisions and actions

NS research directly addresses long-standing obstacles including imperfect or incomplete knowledge, the difficulty of semantic parsing, and computational scaling. NS is oriented toward long-term science via a focused and sequentially constructive research program, with open and collaborative publishing, and periodic spinoff technologies, with a small selection of motivating use cases over time.

The primary ones currently include the pursuit of true natural language understanding via the proxy of question answering;  automatic data science, programming, and mathematics; and financial trading/risk optimization as ways to showcase the fundamental principles being developed. Research in NS is inherently multi-disciplinary and includes (among many other things) work in learning theory and foundations, optimization and algorithms, knowledge representation and acquisition, logic and theorem proving, reinforcement learning, planning, and control, and multi-task/meta/transfer learning.

Neuro-symbolic AI toolkit (NSTK) provide links to all the efforts related to neuro-symbolic AI at IBM Research. Some repositories are grouped together according the meta-projects or pipelines they serve.


## Logical Neural Networks (LNN)

LNNs are a novel `Neuro = symbolic` framework designed to seamlessly provide key properties of both neural nets (learning) and symbolic logic (knowledge and reasoning).

- Every neuron has a meaning as a component of a formula in a weighted real-valued logic, yielding a highly interpretable disentangled representation.
- Inference is omnidirectional rather than focused on predefined target variables, and corresponds to logical reasoning, including classical first-order logic (FOL) theorem proving as a special case.
- The model is end-to-end differentiable, and learning minimizes a novel loss function capturing logical contradiction, yielding resilience to inconsistent knowledge.
- It also enables the open-world assumption by maintaining bounds on truth values which can have probabilistic semantics, yielding resilience to incomplete knowledge.

**Citation:**
```raw
@article{riegel2020logical,
  title={Logical neural networks},
  author={Riegel, Ryan and Gray, Alexander and Luus, Francois and Khan, Naweed and Makondo, Ndivhuwo and Akhalwaya, Ismail Yunus and Qian, Haifeng and Fagin, Ronald and Barahona, Francisco and Sharma, Udit and others},
  journal={arXiv preprint arXiv:2006.13155},
  year={2020}
}
```

| No. | Repository | Main Contributors | Description |
|:---|:---|:---|:---|
| 1    | [FOL-LNN](https://github.com/IBM/LNN) | Naweed Khan, Ndivhuwo Makondo, Francois Luus, Dheeraj Sreedhar, Ismail Akhalwaya, Richard Young, Toby Kurien | First-order logic CPU implementation of LNNs with bounds, downward inference, constraint learning and lifted reasoning in a dynamic graph |
| 2    | [Tensor-LNN](https://github.com/IBM/TensorLNN) | Anamitra Roy Choudhury, Venkatesan Chakaravarthy, Ananda Pal, Yogish Sabharwal | GPU scaling of Propositional LNN for Word Sense Disambiguation and Text Word Common Sense - implementation using sparse-tensors with bounds, downward and an AND/NOT representation |


## Neuro-Symbolic Question Answering (NSQA)
Knowledge base question answering (KBQA) is a task where end-to-end deep learning techniques have faced significant challenges such as the need for semantic parsing, reasoning, and large training datasets. In this work, we demonstrate NSQA, which is a realization of a hybrid "neuro-symbolic" approach. This system integrates multiple, reusable modules that are trained specifically for their individual tasks (e.g.semantic parsing, entity linking, and relationship linking) and require minimal training for the overall KBQA task, therefore addressing the challenges of end-to-end deep learning systems for KBQA.

**Citation:**
```raw
@inproceedings{kapanipathi2021leveraging,
  title={Leveraging abstract meaning representation for knowledge base question answering},
  author={Kapanipathi, Pavan and Abdelaziz, Ibrahim and Ravishankar, Srinivas and Roukos, Salim and Gray, Alexander and Astudillo, Ram{\'o}n Fernandez and Chang, Maria and Cornelio, Cristina and Dana, Saswati and Fokoue-Nkoutche, Achille and others},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={3884--3894},
  year={2021}
}
```

| No. | Repository | Main Contributors | Description |
|:---|:---|:---|:---|
| 1    |  [AMR Parsing](https://github.com/IBM/transition-amr-parser) | Ramon/Young-suk   |   Neural transition-based parser for Abstract Meaning Representation (AMR) producing state-of-the-art AMR parsing and reliable token to node alignments.      |
| 2    |  [Answer Type Prediction](https://github.com/IBM/answer-type-prediction)     | G P Shrivatsa Bhargav    | This system can predict the knowledge base types of the answer to a given natural language question.                                                   |
| 3    |  [SLING](https://github.com/IBM/kbqa-relation-linking)    | Nandana Mihindukulasooriya     |    A relation linking framework which leverages semantic parsing using Abstract Meaning Representation (AMR) and distant supervision. SLING integrates multiple approaches that capture complementary signals such as linguistic cues, rich semantic representation, and information from the knowledge base.                                             |
| 4    |  [Sem Rel](https://github.com/IBM/kbqa-relation-linking) (Part of repo)        | Tahira Naseem    |    A simple transformer-based neural model for relation linking that leverages the AMR semantic parse of a sentence.                                                 |
| 5    |  [GenRel](https://github.com/IBM/kbqa-relation-linking) (Part of repo)        | Gaetano Rossilleo    |   A formulation of relation linking as a generative problem, which facilitates the use of pre-trained sequence-to-sequence models infused with structured data from the target knowledge base.                                                 |
| 6    |  [Logic Embeddings](https://github.com/francoisluus/KGReasoning)                        |  Francois Luus   |    A new approach to embedding complex queries that uses Skolemisation to eliminate existential variables for efficient querying.    |
| 7    |  [TempQA-WD Dataset](https://github.com/IBM/tempqa-wd)                                   |  Sumit Neelam, Udit Sharma, Hima Karanam, Shajith mohamed, Pavan Kapanipathi, Ibrahim Abdelaziz, Nandana Mihindukulasooriya, Srinivas Ravishankar, Maria Chang, Rosario, Achille, Dinesh K, Dinesh G, G P Srivatsa Bhargav,  Saswati Dana    |   Temporal reasoning dataset adapted to wikidata for Natural language question answering over knowledge bases. |                                                 |
| 8    |  [ERGO](https://github.com/IBM/expressive-reasoning-graph-store) | Udit Sharma, Sumit Neelam, Hima Karanam, Shajith mohemad, Achille Fokoue, Ibrahim Abdelaziz    |   Repository for reasoning enabled graph store that supports OWL reasoning. |


## Neuro-Symbolic Agent (NeSA)
Neuro-Symbolic Agent (NeSA) is a neuro-symbolic approach to sequential decision making in real-world problems that is quantitatively and qualitatively better than end-to-end deep learning methods. NeSA uses domain knowledge, commonsense knowledge, and reasoning to reduce the number of interactions with the environment for learning the policy. The policy is represented as Logical Neural Networks (LNN), this enables explainable decision making and co-learning of the policy where the human can guide the training. Unlike, end-to-end system each stage of the pipeline can be individually verified and examined. For more details of the pipeline

**Citation:**
```raw
@inproceedings{kimura-etal-2021-neuro,
    title = "Neuro-Symbolic Reinforcement Learning with First-Order Logic",
    author = "Kimura, Daiki  and  Ono, Masaki  and  Chaudhury, Subhajit  and  Kohita, Ryosuke  and  Wachi, Akifumi  and  Agravante, Don Joven  and  Tatsubori, Michiaki  and  Munawar, Asim  and  Gray, Alexander",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.283",
    doi = "10.18653/v1/2021.emnlp-main.283",
    pages = "3505--3511",
}
```

| No. | Repository | Main Contributors | Description |
|:---|:---|:---|:---|
| 1 | [Logical Optimal Actions (LOA)](https://github.com/IBM/LOA) | Daiki Kimura, Subhajit Chaudhury, Sarathkrishna Swaminathan, Michiaki Tatsubori | LOA is the core of NeSA. It uses reinforcement learning with reward maximization to train the policy as a logical neural network. |
| 2 | [NeSA Demo](https://github.com/IBM/nesa-demo) | Daiki Kimura, Steve Carrow, Stefan Zecevic | This is the HCI component of NeSA. It allows the user to visualize the logical facts, learned policy, accuracy and other metrics. In the future, this will also allow the user to edit the knowledge and the learned policy. It also supports a general purpose visualization and editing tool for any LNN based network. |
| 3 | [TextWorld Commonsense (TWC)](https://github.com/IBM/commonsense-rl) | Keerthiram Murugesan | A room cleaning game based on TextWorld game engine. The game is intractable without the commonsense knowledge about the ususal locations of objects. This is the first task we have solved with NeSA.|
| 4 | [AMR-to-Logic](https://github.com/IBM/AMR-CSLogic) | Vernon Austel, Jason Liang, Rosario Uceda-Sosa, Masaki Ono, Daiki Kimura | Semantic parsing part of the NeSA pipeline to convert natural language text into contextual logic. The logic generated by this component is used by the next stages of the pipeline to learn the policy.  The development repository is [here](https://github.com/CognitiveHorizons/AMR-CSLogic) (limited access). |
| 5 | [CREST](https://github.com/IBM/context-relevant-pruning-textrl) | Subhajit Chaudhury | Repository for EMNLP 2020 paper, Bootstrapped Q-learning with Context Relevant Observation Pruning to Generalize in Text-based Games. `Subhajit Chaudhury, Daiki Kimura, Kartik Talamadupula, Michiaki Tatsubori, Asim Munawar, Ryuki Tachibana. "Bootstrapped Q-learning with Context Relevant Observation Pruning to Generalize in Text-based Games", Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020` |
| 6 | [Logical Twins (Textworld-to-PDDLGym)](https://github.com/IBM/logicaltwins) | Joven Agravante, Michiaki Tatsubori | A simple RL environment wrapper from Textworld to PDDLGym. |

---

## Ethical AI

| No. | Repository | Main Contributors | Description |
|:---|:---|:---|:---|
| 1 |  [Ethical AI Platform](https://github.com/inwonakng/ethical_ai)	| Inwon Kang, Nishant Srivastava, Xinhao Luo, Randolf Xia, Megan Goulet, Ziyi Wang, Joseph Om, Mitesh Kumar, Taras Kaminsky, Norbu Sonam, Shengjin Li	| Website (WIP) for gathering and aggregating ethical preference data. Website only accessible from RPI VPN currently at - https://ethicalai.cs.rpi.edu/ |
| 2 | [Framework for collecting moral preference data](https://github.com/inwonakng/mturk-surveydata-public)	| Inwon Kang, Farhad Mohsin	| Framework for collecting moral preference data with features. Used for collecting two new datasets of preferences under moral dilemma.|
| 3 | [Life Jacket dataset](https://rpi.box.com/s/t8nv9jfe314lac84atpxd8wa6ym27gfy)	| Farhad Mohsin, Inwon Kang	| Life Jacket dataset collected for research regarding learning and aggregating preferences under moral dilemma. Currently have a working manuscript - "Mohsin, F., Kang, I., Chen, P. Y., Rossi, F., & Xia, L. Learning Individual and Collective Priorities over Moral Dilemmas with the Life Jacket Dataset". |
| 4 | [Power plant dataset](https://rpi.box.com/s/k75qvd2v1akdgy97xshfb1cnnp0pg2al)	| Farhad Mohsin, Inwon Kang	| Power plant dataset collected for research regarding learning and aggregating preferences under moral dilemma and considering external constraints |
| 5 | [Designing new fair voting rules](https://github.com/farhadmohsin/LearningToDesign/)	| Farhad Mohsin, Ao Liu	| Code for research regarding designing new fair voting rules with a data-driven approach and adding privacy measures to the mechanism. "Mohsin, F., Liu, A., Chen, P.-Y., Rossi, F., & Xia, L. (2021). Learning to Design Fair and Private Voting Rules [Presented at AI for Social Good workshop at ĲCAI-2021]." Revised version for this paper being prepared for JAIR submission |
| 6 | [Demo](https://github.com/inwonakng/ethical_ai)	| Farhad Mohsin, Inwon Kang	| Demo website (WIP) for designing voting rules based on the techniques from the paper above. Current version accessible at https://inwonakng.github.io/voting-rules-demo/ |
| 7 | [Axiom Verification](https://github.com/farhadmohsin/AxiomVerification)	| Farhad Mohsin, Sikai Ruan, Qishen Han	| Code for research regarding data-driven verification of social choice axiom for voting rules. Work in progress with current working manuscript - "Mohsin, F., Han, Q., Ruan, S., Chen, P.Y., Rossi, F., & Xia, L. Computing Data-Driven Satisfaction of Voting Axioms." |

---

## A Hypergraph-based Framework for Knowledge Graph Federation and Multimodal Integration

| No. | Repository | Main Contributors | Description |
|:---|:---|:---|:---|
| 1 | [HKLIB](https://github.com/ibm-hyperknowledge/hklib) | Marcio Ferreira Moreno  | A NodeJS library to provide software abstractions for accessing hyperknowledge graphs |

---

## Advances in Neuro-Symbolic AI

| No. | Repository | Main Contributors | Description |
|:---|:---|:---|:---|
| 1 |  [Binary Matrix Factorization](https://github.com/IBM/binary-matrix-factorization)  | Francisco Barahona, Joao Goncalves  | Implementation of algorithms for binary matrix factorization, which consists of approximating a binary matrix with the product of two low rank binary matrices.  |
| 2 |  [Online Alternating Minimization](https://github.com/IBM/online-alt-min)  | Choromanska, Cowen, Kumaravel, Luss, Rigotti,  Rish, Diachille, Gurev, Kingsbury, Tejwani, Bouneffouf  | Code implementing a novel online alternating minimization (AM) algorithm for training deep neural networks that was developed as an alternative to backpropagation to avoid vanishing and exploding gradients, and handle non-differentiable nonlinearities. |
| 3 | [Sobolev Independence Criterion](https://github.com/IBM/SIC) | Mroueh, Sercu, Rigotti, Padhi, dos Santos | Implemention of the Sobolev Independence Criterion (SIC), an interpretable dependency measure that provides feature importance scores and hence can be used for nonlinear feature selection. |
| 4 | [IT Operations Ontology](https://github.com/IBM/ITOPS-ontology) | R. Uceda-Sosa, N. Mihindukulasooriya, S. Bansal, S. Nagar. A. Kumar. V. Agarwal | ITOPS leverages Linked Open Data (LOD) resources like Wikidata to automatically construct domain-specific ontologies. From a small set of user-defined seed concepts, the ITOPS pipeline generates a rich graph that can be queried and inferenced over with standard RDF/OWL technologies. | 
| 5 | [Policy Gradient Algorithm for Learning to Learn in Multiagent RL](https://github.com/dkkim93/meta-mapg) | | Source code for "A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning" (ICML 2021), Publication: http://proceedings.mlr.press/v139/kim21g/kim21g.pdf |
| 6 | [Word embeddings from ontologies](https://github.com/iesl/geometric_graph_embedding) | Michael Boratko, Dongxu Zhang, Nicholas Monath, Luke Vilnis, Kenneth L. Clarkson, Andrew McCallum | Future home of NeurIPS accepted paper "Capacity and Bias of Learned Geometric Embeddings for Directed Graphs" |
| 7 | [Logical Formula Embedder (part of TRAIL)](https://github.com/IBM/LogicalFormulaEmbedder) | Maxwell Crouse, Ibrahim Abdelaziz, Achille Fokoue | Repository for GNNs work of logical formulae with subgraph pooling |
| 8 | [Knowledge-Enabled Textual-Entailment](https://github.com/IBM/knowledge-enabled-textual-entailment) | Pavan Kapanipathi | Natural Language Inference is fundamental to many Natural Language Processing applications such as semantic search and question answering. The task of NLI has gained significant attention in the recent times due to the release of fairly large scale, challenging datasets. Present approaches that address NLI are largely focused on learning based on the given text in order to classify whether the given premise entails, contradicts, or is neutral to the given hypothesis. On the other hand, techniques for Inference, as a central topic in artificial intelligence, has had knowledge bases playing an important role, in particular for formal reasoning tasks. While, there are many open knowledge bases that comprise of various types of information, their use for natural language inference has not been well explored. In this work, we present a simple technique that can harnesses knowledge bases, provided in the form of a graph, for natural language inference. |
| 9 | [Persistence Homology for Link Prediction: An Interactive View](https://github.com/pkuyzy/TLC-GNN) | Zuoyu Yan, Tengfei Ma | Code for paper Link Prediction with Persistent Homology: An Interactive View (ICML2021). In this work, we propose a novel topological approach based on the extended persistent homology to characterize interactions between two nodes. We propose a graph neural network method combining with this topological feature and it outperforms state-of-the-arts on different benchmarks. As another contribution, we propose a novel algorithm to more efficiently compute the extended persistence diagrams for graphs. This algorithm can be generally applied to accelerate many other topological methods for graph learning tasks. |
| 10 | [Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport](https://github.com/matenure/OTCoarsening) | Tengfei Ma | Source codes are for the paper "Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport" (AAAI 2021). The codes are built on the libarary of [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric). This paper proposed a new unsupervised graph learning method based on optimal transport and graph coarsening. |
| 11 | [Reinforcement Learning with Algorithms from Probabilistic Structure Estimation (RLAPSE)](https://github.com/roman1e2f5p8s/rlapseingym) | Djallel Bouneffouf | This repo contain the code for an orchestrator that decide which RL algorithm to use depending on the environment {Bandit, contextual Bandit and Reinforcement learning} |
| 12 | [Meta-Experience Replay (MER)](https://github.com/mattriemer/MER) | Matt Riemer | Source code for the paper "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference" https://openreview.net/pdf?id=B1gTShAct7 |
| 13 | [Formal ML](https://github.com/ibm/formalml) | Vasily Pestun, Nathan Fulton (AWS), Barry Trager, Avi Shinar, Alexander Rademaker | Formalization of Machine Learning Theory with Applications to Program Synthesis. Partial formalization of key results from https://arxiv.org/abs/1804.07795. The CertRL library as reported in https://arxiv.org/abs/2009.11403 |
| 14 | [TM-GCN](https://github.com/IBM/TM-GCN) | Shashanka Ubaru | Pytorch code for TM-GCN method, a Dynamic Graph Convolutional Networks Using the Tensor M-Product. PNAS paper (https://www.pnas.org/content/118/28/e2015851118.short), SIAM paper (https://epubs.siam.org/doi/abs/10.1137/1.9781611976700.82), IBM blog (https://research.ibm.com/blog/new-tensor-algebra) |
| 15 | [GraphSEIR_aPCE](https://github.com/Shashankaubaru/GraphSEIR_aPCE) | Shashanka Ubaru, Lior Horesh | Matlab code for Dynamic graph based epidemiological model for COVID-19 contact tracing data analysis and optimal testing prescription. JBI paper (https://www.sciencedirect.com/science/article/pii/S1532046421002306), IBM blog (https://research.ibm.com/blog/accelerating-covid-discoveries) |
| 16 | [Neural Unification for Logic Reasoning over Language](https://github.com/IBM/Neural_Unification_for_Logic_Reasoning_over_Language) | Gabriele Picco, Hoang Thanh Lam, Marco Luca Sbodio, Vanessa Lopez Garcia | In this work we propose a transformer-based architecture, namely the Neural Unifier, and a relative training procedure, for deriving conjectures given axioms expressed in natural language (English). The method achieves state-of-the-art results in term of generalisation on the considered benchmark datasets, showing that mimicking a well-known inference procedure, the backward chaining, it is possible to answer deep queries even when the model is trained only on shallow ones. More information can be found in the full paper: https://aclanthology.org/2021.findings-emnlp.331.pdf |
| 17 | [E-PDDL](https://github.com/FrancescoFabiano/E-PDDL) | Francisco Fabiano (University of Udine) | E-PDDL: A Standardized Way of Defining Epistemic Planning Problems, ICAPS 21 paper: https://arxiv.org/abs/2107.08739 |

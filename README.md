# Multimodal classifier-induced stopping (CIS)
Base code for 'Early Classifying Multimodal Sequences' by Alexander Cao, Jean Utke, and Diego Klabjan (ICMI 2023)

Two top-level folders correpsond with the first two experiments from the paper. Please see the paper for data sources. Within each experiment's folder:
-  data folder: code for splitting data into train, validation, and test sets
-  peripherals: code for training peripheals/extracting features from each modality
-  larm: code for LARM benchmark ($\mu=10^{-5}$ for run 1)
-  cis: code for proposed CIS method ($\mu=10^{-5}$ fo run 1)
-  paretoAUC: code for plotting Pareto frontiers of above two methods (multiple mu) and calculating AUC
-  histT (for esp): code for plotting Figure 4 (bottom) in the paper i.e. human interpretation of results

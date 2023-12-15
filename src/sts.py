import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import spacy
nlp = spacy.load("en_core_web_sm")

class STSAnalyzer:
    
    INPUT_FOLDER = "input"
    TRAIN_PATH = os.path.join(INPUT_FOLDER, "train")
    TEST_PATH = os.path.join(INPUT_FOLDER, "test")
    LS_FILES_TRAIN = ["MSRpar", "MSRvid", "SMTeuroparl"]
    LS_FILES_TEST = ["MSRpar", "MSRvid", "SMTeuroparl", "surprise.SMTnews", "surprise.OnWN"]

    def __init__(self, 
                 preprocess_fun, 
                 model,
                 weights=None,
                 input_path=TEST_PATH,
                 ls_files=LS_FILES_TEST,
                 name=None):
        
        '''
        Input:
            preprocess_fun - function that preprocesses a string and returns a set of words
            model - model that takes as input a set of words and returns a similarity score
            weights - weights for allnorm measure (default None)
            input_path - path to the input files (default TEST_PATH)
            ls_files - list of files to be used (default LS_FILES_TEST)
            name - name of the model (default None)
        '''
        self.preprocess_fun = preprocess_fun
        self.name = name
        self.model = model
        self.input_path = input_path
        self.ls_files = ls_files
        self.weights = weights

    def read_file(self, filename):
        '''
         Read the file and return a dataframe with the sentences and the gold standard
        '''
        dt = pd.read_csv(os.path.join(self.input_path, 'STS.input.' + filename + '.txt'), 
                         sep='\t|\\t', header=None, on_bad_lines='skip', engine='python')
        dt['gs'] = pd.read_csv(os.path.join(self.input_path, 'STS.gs.' + filename + '.txt'), 
                               sep='\t|\\t',header=None, on_bad_lines='skip', engine='python')
        dt.columns = ['sent1', 'sent2', 'gs']
        return dt
    
    def load_data(self, ls_files=None):
        '''
        Load the data from the files

        If no files are specified, the initialization files are used
        '''

        if ls_files is None:
            ls_files = self.ls_files

        data = pd.DataFrame()
        for file in ls_files:
            data_file = self.read_file(file)
            data_file['file'] = file
            data = pd.concat([data, data_file], axis=0, ignore_index=True)
        
        self.data = data
        return data

    def get_similarity(self, X):
        '''
        Get the similarity between two sentences
        '''

        #### Here we can create the features for the model    
        #### or if it's computationally expensive, we can create them 
        #### in the load_data function or load it from a file
        
        # Return the similarity between the two sets of words
        return self.model.predict(X)
    
    def corr_with_gold(self, filename):
        '''
        Returns the correlation between the gold standard and the similarity
        Input: filename - string with the name of the file
        Output: Pearson correlation object
        '''

        dt = self.data[self.data.file == filename].copy()

        # Get the similarity between the two sentences
        arr_sim = self.get_similarity(dt.iloc[:, :2])
        
        return pearsonr(arr_sim, dt['gs'].values)

    def ind_report(self, file, n_show=10):
        '''
        Show the difference between the gold standard and the similarity
        and the intersection and difference between the two sets of words

        The objective is to understand why the model is failing
        '''

        print("=====================================================")
        print(file)
        print("=====================================================")
        print("\n")

        # Load the data, calculate the similarity and the difference with the gold standard
        df_ind = self.data[self.data.file == file].copy()
        df_ind['sim'] = self.get_similarity(df_ind.iloc[:, :2])
        df_ind['dif'] = df_ind['sim'] - df_ind['gs']

        # Show how many of the "worst" sentences you want to see
        if n_show is not None:
            difs_tail = df_ind.sort_values(by='dif', ascending=False).tail(n_show)
        else:
            difs_tail = df_ind.sort_values(by='dif', ascending=False)
        
        # Show the sentences and the difference between the gold standard and the similarity
        for _, row in difs_tail.iterrows():
            x, y = row.iloc[0], row.iloc[1]
            prep_x, prep_y = self.preprocess_fun(x, out_set=True), self.preprocess_fun(y, out_set=True)
            print(f"RAW TEXT (DIFFERENCE {row['dif']:.2f})")
            print(x, '\n', y)
            print("\nPREPROCESSED TEXT")
            print(prep_x, '\n', prep_y)
            print("INTERSECTION", prep_x.intersection(prep_y))
            print("DIFFERENCE", prep_x.difference(prep_y).union(prep_y.difference(prep_x)))
            print('\n\n')

        # Show the histogram and boxplot of the difference between the gold standard and the similarity
        _, ax = plt.subplots(1, 2, figsize=(10,2.5))
        plt.suptitle(file)
        df_ind.dif.plot.hist(bins=20, title="Residuals", ax=ax[0])
        df_ind.dif.plot.box(title="Residuals", ax=ax[1]);
    
    def col_report(self, n_show=10):
        '''
        Show the difference between the gold standard and the similarity
        and the intersection and difference between the two sets of words
        for all the files
        '''

        for file in self.ls_files:
            self.ind_report(file, n_show=n_show)
        
    def corr_table(self, ls_files):
        '''
        Calculate the Pearson correlation between the gold standard and the similarity
        '''
        df_scores = pd.DataFrame(columns=['file','pearson','p_value', 'conf_low', 'conf_high'])

        ls_corr = []
        ls_p_value = []
        ls_conf_low, ls_conf_high = [], []

        for file in ls_files:
            print(f"Processing {file}")
            pearson_stats = self.corr_with_gold(file)
            ls_corr.append(pearson_stats.statistic)
            ls_p_value.append(pearson_stats.pvalue)
            ls_conf_low.append(pearson_stats.confidence_interval(.95).low)
            ls_conf_high.append(pearson_stats.confidence_interval(.95).high)

        df_scores['file'] = ls_files
        df_scores['pearson'] = ls_corr
        df_scores['p_value'] = ls_p_value
        df_scores['conf_low'] = ls_conf_low
        df_scores['conf_high'] = ls_conf_high
        
        if self.name is not None:
            df_scores['name'] = self.name

        self.df_scores = df_scores
        return df_scores
    
    def get_results(self):
        '''
        Calculate the 3 evaluation measures:
            - Pearson correlation for the concatenation of all five datasets
            - Autoscaled correlation (ALLNORM)
            - Weighted mean correlation (WMEAN)
        '''

        # Get the data and calculate the similarity
        df_col_analysis = self.data.copy()
        df_col_analysis['sim'] = self.get_similarity(df_col_analysis.iloc[:, :2])

        # Pearson correlation for the concatenation of all datasets
        self.total_corr = pearsonr(df_col_analysis['sim'].values, df_col_analysis['gs'].values)

        # Autoscaled correlation (ALLNORM)
        # reg = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=3)
        # arr_preds = np.concatenate(df_col_analysis.groupby('file').\
        #                             apply(lambda x: reg.fit(x.sim.values.reshape(-1, 1), x.gs.values).predict(x.sim.values.reshape(-1, 1))).values)
        # self.allnorm = pearsonr(arr_preds, df_col_analysis['gs'].values)
    

        # Weighted mean correlation (WMEAN)
        self.w_mean = df_col_analysis.groupby('file').\
                        apply(lambda x: pearsonr(x.sim.values, x.gs.values).statistic * x.shape[0] / df_col_analysis.shape[0]).sum()
        
        # Weighted mean of the Pearson correlations on individual datasets
        df_global_scores = pd.DataFrame(columns=['all', 'wmean']) # 'allnorm'
        df_global_scores.loc[0] = [self.total_corr[0], self.w_mean] # self.allnorm[0]

        if self.name is not None:
            df_global_scores['name'] = self.name
            
        self.df_global_scores = df_global_scores
        # self.df_col_analysis = df_col_analysis

        return df_global_scores
    
    def plot_col_analysis(self, ls_sts=[]):
        '''
        Plot the results and compare with other models

        Input:
            ls_sts - list of STSAnalyzer objects to compare with
        '''

        # if self.df_col_analysis is None:
        #     self.
        # df_scores_comp = self.df_scores.copy()
        df_global_scores_comp = self.df_global_scores.copy()

        # Concatenate the results of the other models
        if ls_sts != []:
            for sts in ls_sts:
                # df_scores_comp = pd.concat([df_scores_comp, sts.df_scores], 
                #                              axis=0, ignore_index=True)
                df_global_scores_comp = pd.concat([df_global_scores_comp, sts.df_global_scores],
                                                     axis=0, ignore_index=True)
                
        # plt.figure(figsize=(10,5))
        # sns.barplot(x='file', y='pearson', hue='name', data=df_scores_comp)
        # plt.title('Pearson correlation between similarity and gold standard for different preprocessings')
        # plt.xticks(rotation=45)

        # Plot them
        df_global_comp_plot = df_global_scores_comp.melt(id_vars='name', value_vars=['all', 'wmean']) # 'allnorm',
        plt.figure(figsize=(10,5))
        sns.lineplot(x="variable", y="value", hue="name", data=df_global_comp_plot, marker='o')
        if df_global_comp_plot.value .abs().min()> 0.3:
            plt.ylim(0.5, 1.);


        
                        
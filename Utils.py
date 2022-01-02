import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from scipy.stats import norm
import time


class PlotUtils:
    @staticmethod
    def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
        title = 'Confusion Matrix of {}'.format(title)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def plot_heatmap(dataframe):
        time.sleep(1)
        fig = plt.figure(figsize = (15, 15))
        plt.title('Heatmap of Correlation')  
        sns.heatmap(dataframe.corr(), cmap = "coolwarm", annot = True, fmt = ".2f", annot_kws = {"fontsize": 9},
                    vmin = -1, vmax = 1, square = True, linewidths = 0.01, linecolor = "black", cbar = True)
        sns.despine(top = True, right = True, left = True, bottom = True)
        # st.pyplot(fig)
        plt.show()

    @staticmethod
    def plot_CountByClass(dataframe):
        time.sleep(1)
        fig = plt.figure(figsize = (12,5),dpi = 100)
        plt.subplot(1,2,1)
        ax = sns.countplot('Class',data = dataframe)
        # ax.set_yscale('log')
        plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions')
        plt.subplot(1,2,2)
        countdata = [dataframe[dataframe["Class"] == 0]["Class"].count(),dataframe[dataframe["Class"] == 1]["Class"].count()]
        labelsdata = ["Normal","Fraud"]
        colors = sns.color_palette('pastel')[0:2]
        plt.pie(countdata,labels=labelsdata,autopct='%.2f%%')
        plt.title("The Ratio between two classes")
        plt.show()
        # st.pyplot(fig)

    @staticmethod
    def plot_distributionByfeature(dataframe, feature):
        time.sleep(1)
        DATA = dataframe.copy()
        DATA['hour'] = DATA[feature].apply(lambda x: np.ceil(float(x)/3600) % 24)
        timedata = pd.concat([shuffle(DATA[DATA["Class"] == 0 ]).iloc[:500],  DATA[DATA["Class"] == 1]])
        sns.set_style("white")
        bins = np.arange(timedata['hour'].min(),timedata['hour'].max()+2)
        fig = plt.figure(figsize=(15,5))
        # plot a distribution plot according to the hour
        sns.distplot(timedata[timedata['Class']==0.0]['hour'],bins=bins,kde=True,hist_kws={'alpha':.5}, label='Normal')
        sns.distplot(timedata[timedata['Class']==1.0]['hour'],bins=bins,kde=True,label='Fraud',hist_kws={'alpha':.5})
        plt.xticks(range(0,24))
        plt.legend()
        plt.title("The distribution of amount according to hour")
        plt.xlabel("Hour")
        plt.ylabel("Density")
        plt.show()
        # st.pyplot(fig)  

    @staticmethod
    def plot_single_distribution(dataframe,feature):
        time.sleep(1)
        fig = plt.figure(figsize=(10,8))
        plt.title('Distribution of ' +feature+ ' Feature')
        sns.distplot(dataframe[feature])
        # st.pyplot(fig)
        plt.show()

    @staticmethod
    def plot_multipe_distributions(dataframe, features, c):
        time.sleep(1)
        fig, axes = plt.subplots(1,3, figsize=(20, 6))
        colors = ['#FB8861','#56F9BB', '#C5B3F9']
        color_index = 0
        for index,feature in enumerate(features):
            class_dist = dataframe[feature].loc[dataframe['Class'] == c].values
            sns.distplot(class_dist,ax=axes[index], fit=norm, color=colors[color_index])
            if c==1 :
                class_name = "Fraud"
            else :
                 class_name = "Clear"
            axes[index].set_title(feature + ' Distribution \n ('+class_name+' Transactions)', fontsize=14)
            if color_index == 2:
                color_index = 0
            else:
                color_index = color_index+1
        # st.pyplot(fig)
        plt.show()

    @staticmethod
    def plot_all_distributions(dataframe):
        time.sleep(1)
        features = dataframe.columns.values
        t0 = dataframe.loc[dataframe['Class'] == 0]
        t1 = dataframe.loc[dataframe['Class'] == 1]
        fig = plt.figure(figsize = (20,25))
        i=0
        for feature in features:
            sns.set_style('ticks')
            i += 1
            plt.subplot(8,4,i)
            sns.kdeplot(t0[feature],bw=0.5,label="Normal",shade= True,alpha = 0.3)
            sns.kdeplot(t1[feature],bw=0.5,label="Fraud",shade= True,alpha = 0.3)
            plt.xlabel(feature, fontsize=12)
            plt.xticks()
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.legend(fontsize = "medium",loc = "best")
            plt.suptitle('The data distribution of different features',y = 0.9,fontsize = 20)
        plt.show()
        # st.pyplot(fig)
        # st.info("From the distribution plot, we could obviously found that the variance of the Fraud part of the dataset is much larger than the normal part, lefr or right skewed than the normal part.")

    @staticmethod
    def plot_two_features_classification(dataframe, target, f1, f2):
        time.sleep(1)
        fig, ax = plt.subplots(figsize=(15, 8))
        # ax.set_facecolor("#393838")
        X = dataframe.drop(target, axis = 1)
        y = dataframe[target].values
        labels = dataframe[target].value_counts().index.tolist()
        ax.scatter(X.loc[y == 0, f1], X.loc[y == 0, f2], label = labels[0], alpha = 1, linewidth = 0, c = "#0EB8F1")
        ax.scatter(X.loc[y == 1, f1], X.loc[y == 1, f2], label = labels[1], alpha = 1, linewidth = 0, c = '#F1480F', marker = "X")
        ax.set_title("Distribution of " + target + " w.r.t " + f1 + " and " + f2)
        ax.set_xlabel(f1); ax.set_ylabel(f2)
        ax.legend()
        sns.despine(top = True, right = True, left = True, bottom = True)
        plt.show()
        # st.pyplot(fig)

    @staticmethod
    def plot_distribution_box(dataframe):
        time.sleep(1)
        fig, ax = plt.subplots(figsize=(16,6))
        plt.style.use('ggplot') # Using ggplot2 style visuals 
        ax.set_facecolor('#fafafa')
        ax = sns.boxplot(data = dataframe, palette = 'Set3',whis = 2.5)
        plt.title("The box distribution of V1 - V28")
        plt.show()
        # st.pyplot(fig)

    @staticmethod
    def plot_multipe_distribution_box(dataframe, features):
        time.sleep(1)
        fig, axes = plt.subplots(ncols=len(features), figsize=(20,6))
        # Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
        for index,feature in enumerate(features):
            sns.boxplot(x="Class", y=feature, data=dataframe, palette=sns.color_palette('pastel')[0:2], ax=axes[index])
            axes[index].set_title(feature + ' vs Class Positive Correlation')
        plt.show()
        # st.pyplot(fig)

class AlgoUtils:

    @staticmethod
    def min_max_normalization():
        return 
    
    @staticmethod
    def random_under_simpling():
        return 0
    
    @staticmethod
    def detect_outliers():
        return 0



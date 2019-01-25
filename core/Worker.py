import pandas as pd
import numpy as np

from IPython.display import display, clear_output
from sklearn.externals import joblib
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import time
import datetime
import os
from itertools import zip_longest
import warnings
import re

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# from Codes_helper import Codes_helper
from .Codes_helper import Codes_helper


class Worker():
    def __init__(self, 
                 w2v_model  = None, 
                 w2v_size   = None, 
                 lang       = None, 
                 conv_type  = None, 
                 rubr_id    = None, 
                 clf        = None, 
                 data_train = None, 
                 data_test  = None, 
                 name_train = None, 
                 name_test  = None, 
                 res_folder = None,
                 clear_math = True):
        self.w2v_model  = w2v_model
        self.w2v_size   = w2v_size
        self.lang       = lang
        self.conv_type  = conv_type
        self.rubr_id    = rubr_id
        self.clf        = clf
        self.data_train = data_train
        self.data_test  = data_test
        self.name_train = name_train
        self.name_test  = name_test
        self.res_folder = res_folder
        self.clear_math = clear_math
        
    def load_w2v(self, w2v_path):
        """
        Loads word2vec model.

        Args:
        w2v_path (str): absolute or relative path to the w2v model file.
        """
        if os.path.exists(w2v_path):
            self.w2v_model = Word2Vec.load(w2v_path)
            if     '_ru'  in w2v_path:
                self.lang = 'ru'
            elif   '_en'  in w2v_path:
                self.lang = 'en'
            self.w2v_size = self.w2v_model.layer1_size
            return True
        else:
            print('Path to word2vec model is not valid.')
            return False
    
    def load_clf(self, clf_path):
        """
        Loads claffifier from given file name.

        Args:
        clf_path (str): absolute or relative path to the classifier model file.
        """
        if os.path.exists(clf_path):
            if   '_sum'  in clf_path:
                self.conv_type = 'sum'
            elif '_max'  in clf_path:
                self.conv_type = 'max'
            elif '_mean' in clf_path:
                self.conv_type = 'mean'
            if   'ipv'   in  clf_path:
                self.rubr_id = 'ipv'
            elif 'subj'  in  clf_path:
                self.rubr_id = 'subj'
            elif 'rgnti' in  clf_path:
                self.rubr_id = 'rgnti'
            self.clf = joblib.load(clf_path)
            return True
        else:
            return False
    
    def load_data(self, train_path, test_path=None, split_ratio=0.8, sep='\t'):
        """
        Loads test and train data. If test_path is equal to None then train set will be splitted into two parts 
        according to split_ratio. 
        Loaded data in not transformed and contains all rubrics and additional fields.

        Args:
            train_path (str): absolute or relative path to train data file.
            test_path (str): absolute or relative path to test data file. If is equal to None, train data 
                           will be separated to make train and test sets.
            split_ratio (int): proportion of train data in splitting.
            
        Returns:
            True or False depending on the success of the task.
        """
        if train_path and os.path.exists(train_path):
            self.name_train = train_path
            if test_path and os.path.exists(test_path):
                self.name_test = test_path
                self.data_train = pd.read_csv(train_path, index_col=0, sep=sep)
                self.data_test = pd.read_csv(test_path, index_col=0, sep=sep)
            else:
                if test_path:
                    print('Test path is not valid, train set will be splitted.')
                self.name_test = train_path
                data = pd.read_csv(train_path, index_col=0, sep=sep)
                train_index, test_index = train_test_split(data.index.unique(), 
                                                           test_size=1-split_ratio)
                self.data_train, self.data_test = data.loc[train_index], data.loc[test_index]
            if   '_sum'  in os.path.split(train_path)[-1]:
                self.conv_type = 'sum'
            elif '_max'  in os.path.split(train_path)[-1]:
                self.conv_type = 'max'
            elif '_mean' in os.path.split(train_path)[-1]:
                self.conv_type = 'mean'
            if "_ru" in os.path.split(train_path)[-1]:
                self.set_lang("ru")
            elif "_en" in os.path.split(train_path)[-1]:
                self.set_lang("en")
            s = re.findall('(?<=sum|ean|max)(\d+)', train_path)
            if s:
                self.w2v_size = int(s[0])
            return True
        else:
            print('Please specify existing train data path.')
            self.data_train = None
            self.data_test = None
            return False
        
    def data_cleaning(self, description=None):
        """
        Creates one or two files with train and test data with only one rubric per string.
        """
        if self.data_train is None:
            print("Please load raw data to train or train and test fields in Worker object")
            return False
        if not (self.data_test is None):
            self.data_test = self.__split_all_sect(self.data_test)
            if description:
                d = 'test_single_theme'+'_'+description
            else:
                d = 'test_single_theme'
            test_name = self.__create_name('data', self.data_test, description=d)
            self.__save_file(test_name, self.data_test)
            self.name_test = test_name
        self.data_train = self.__split_all_sect(self.data_train)
        if description:
            d = 'single_theme'+'_'+description
        else:
            d = 'single_theme'        
        train_name = self.__create_name('data', self.data_train, description=d)
        self.__save_file(train_name, self.data_train)
        self.name_train = train_name
        return True

    def set_res_folder(self, path):
        """
        Creates directory for saving current working files.

        Args:
        path (str): absolute or relative path to result folder. If folder does not exixts it will be created. 
                       In that case head of path must exists.
        """
        if os.path.exists(path) and os.path.isdir(path):
            self.res_folder = path
            return True
        elif os.path.exists(os.path.split(path)[0]):
            os.makedirs(path)
            self.res_folder = path
            return True
        else:
            print('Result directory creation failed.')
            self.res_folder = None
            return False
    
    def set_rubr_id(self, rubric):
        """
        Checkes the correctness of input data and sets rubric if it is fine.
        
        Args:
        rubric (str): name of rubric to be set.
        """
        rubric = rubric.lower()
        if rubric in ['ipv', 'subj', 'rgnti']:
            self.rubr_id = rubric
            return True
        else:
            print('Not a valid rubric name. Please choose one of "ipv", "subj", "rgnti".')
            return False
    
    def set_conv_type(self, conv_type):
        """
        Checkes the correctness of input data and sets convlution type if it is fine.
        
        Args:
        conv_type (str): name of convolution type to be set.
        """
        conv_type = conv_type.lower()
        if conv_type in ['mean', 'max', 'sum']:
            self.conv_type = conv_type
        else:
            print('Not a valid convolution type name. Please choose one of "mean", "max", "sum".')
    
    def set_clf(self, path):
        """
        Checkes the correctness of input path and setas classification model if it is fine.
        
        Args:
        path (str): absolute or relative path to classification model.
        """
        if os.path.exists(path) and path[-4:] == '.pkl':
            self.load_clf(path)
            return True
        else:
            print('Not a valid path, try again. File type should be ".pkl".')
            return False
    
    def set_w2v(self, path):
        """
        Checkes the correctness of input path and setas word2vec model if it is fine.
        
        Args:
        path (str): absolute or relative path to word2vec model.
        """
        if os.path.exists(path) and path[-6:] == '.model':
            self.load_w2v(path)
            return True
        else:
            print('Not a valid path, try again. File type should be ".model".')
            return False
        
    ################################################
    def set_lang(self, lang):
        if lang in ['ru', 'en']:
            self.lang = lang
            return True
        else:
            print('Not a valid language. Please choose "en" or "ru".')
            return False

    def set_math(self, math:bool):
        self.math = math
    ################################################

    def __check_res_folder(self):
        """
        Checkes if result folder is set. 
        If it is not, function asks a new path and checks if it is correct.
        """
        while not self.res_folder:
            print('Please specify result folder:')
            folder = input()
            if os.path.exists(folder):
                self.set_res_folder(folder)
            else:
                print('Not a valid path, try again.')
        return True
           
    def __check_rubr_id(self):
        """
        Checkes if rubric is set. 
        If it is not, function asks a new rubric and checks if it is correct.
        """
        while self.rubr_id is None:
            print('Please specify rubric id:')
            rubric = input()
            self.set_rubr_id(rubric)
        return True

    def __check_conv_type(self):
        """
        Checkes if convolution type is set. 
        If it is not, function asks a new convolution type and checks if it is correct.
        """
        while self.conv_type is None:
            print('Please specify convolution type:')
            conv_type = input()
            self.set_conv_type(conv_type)
        return True
    
    def __check_clf(self):
        """
        Checkes if classifier is set. 
        If it is not, function asks a new classifier path and checks if it is correct.
        """
        while self.clf is None:
            print('Please specify path to classifier file:')
            file = input()
            self.set_clf(file)
        
    def __check_w2v(self):
        """
        Checkes if word2vec model is set. 
        If it is not, function asks a new word2vec path and checks if it is correct.
        """
        while self.w2v_model is None:
            print('Please specify path to word2vec model file:')
            file = input()
            self.set_w2v(file)
    ####################################################
    
    def __check_lang(self):
        while not self.lang:
            print('Please specify language:')
            lang = input()
            self.set_lang(lang)

    # ??? WTF тут происходит
    def __check_data(self):
        while self.data_train is None:
            self.name_train = None
            while not os.path.exists(file_train):
                print('Please specify path to train data file:')
                file_train = input()
            self.name_test = None
            while file_test and not os.path.exists(file_test):
                print('Please specify path to test data file (leave empty if no such file):')
                file_test = input()
                split = None
                if not file_test:
                    file_test = None
                    print('Please specify split fraction(leave empty if no split):')
                    split = float(input().replace(',', '.'))
            self.load_data(train_path=file_train, test_path=file_test, split_ratio=split)
    ####################################################
    
    def __create_name(self, file_type, save_data, version=1, description=None, info=None):
        """
        Creates saving name for file in result_folder. Always add "test" in description 
            for use it's features in name. By default train dataset is used.

        Args:
        file_type (str): type of data that file contains, can be "data", "clf_model", 
                       "result", "answers", "w2v_model", "w2v_vectors"
        save_data   -- string/DataFrame/dict{int or str:pd.DataFrame}/w2v model/clf model for saving.
        version     -- int or str with version of file.
        description (str): additional data that should be in file name.
        info        -- if file contains additional information or not (bool).
        """
        self.__check_res_folder()
        if self.data_test is not None:
            test_amount = self.data_test.index.drop_duplicates().shape[0]/1000
        else:
            test_amount = '-'
        if self.data_train is not None:
            train_amount = self.data_train.index.drop_duplicates().shape[0]/1000
        else:
            train_amount = '-'
        if description:
            description = str(description)
        name = self.res_folder + '/'
        if info is not None:
            name += 'info_'
        name += file_type
        
        if file_type == 'w2v_model':
            name += '_'+str(self.w2v_size)
            name += '_'+self.lang
            name += '_'+str(round(test_amount+train_amount))+'k'
            if description:
                name += '_'+description
        elif file_type == 'clf_model':
            name += '_'+self.lang
            name += '_'+self.rubr_id
            if description:
                name += '_'+description
            name += '_'+self.conv_type
            name += str(self.w2v_size)
            
        elif file_type == 'data':
            if description:
                name += '_'+description
            if self.lang is not None:
                name += '_'+self.lang                
            if 'test' in description:
                name += '_'+str(round(test_amount))+'k'
            else:
                name += '_'+str(round(train_amount))+'k'
            
        elif file_type == 'w2v_vectors':
            if description is None:
                name += '_'+str(round(train_amount))+'k'
            else:
                name += '_'+description
                if 'test' in description:
                    name += '_'+str(round(test_amount))+'k'
                else:
                    name += '_'+str(round(train_amount))+'k'
            name += '_'+self.conv_type
            name += str(self.w2v_size)
        
        elif file_type == 'answers':
            name += '_'+self.rubr_id
            name += '_'+self.lang
            name += '_'+self.conv_type
            name += str(self.w2v_size) 
            if 'test' in description:
                name += '_'+str(round(test_amount))+'k'
            else:
                name += '_'+str(round(train_amount))+'k'
            
        elif file_type == 'result':
            name += '_'+self.rubr_id
            name += '_'+self.lang
            name += '_'+self.conv_type
            name += str(self.w2v_size)
            if description and 'test' in description:
                name += '_'+str(round(test_amount))+'k'
            else:
                name += '_'+str(round(train_amount))+'k'
        
        name += '_v'+str(version)
        now = datetime.datetime.today()
        date = str(now.day)+'_'+str(now.month)+'_'+str(now.year)[2:]
        name += '_'+date
        
        if type(save_data)   == str:
            name += '.txt'     
        elif type(save_data) == pd.core.frame.DataFrame:
            name += '.csv'
        elif type(save_data) == Word2Vec:
            name += '.model'
        elif type(save_data) == dict:
            name += '.xlsx'
        else:
            name += '.plk'
        return name
    
    def create_sets(self,
                    path_ipv_codes='./RJ_code_21017_utf8.txt',
                    path_replacement='./Replacement_RJ_code_utf8.txt',
                    split_ratio=None):
        """
        Creates clear train and test X and y based on current train and test sets in object.
        Needs right names of chosen rubric column according to format ("subj", "ipv", "rgnti").

        Args:
        split_ratio (int): needed if there ae no test data specified in self.data_test.
        """
        self.__check_rubr_id()
        helper = Codes_helper(clear_math=not(self.math))
        if self.rubr_id == 'ipv':
            helper.set_ipv_codes(path_ipv_codes)
            helper.set_ipv_change(path_replacement)
            if self.data_train is not None:
                self.data_train = helper.change_ipv(self.data_train)
            if self.data_test is not None:
                self.data_test = helper.change_ipv(self.data_test)
        elif self.rubr_id == 'subj':
            if self.data_train is not None:
                self.data_train = helper.change_subj(self.data_train)
            if self.data_test is not None:
                self.data_test = helper.change_subj(self.data_test)
        elif self.rubr_id == 'rgnti':
            if self.data_train is not None:
                self.data_train = helper.change_rgnti(self.data_train)
            if self.data_test is not None:
                self.data_test = helper.change_rgnti(self.data_test)

        if split_ratio is None:
            if self.data_train is None and self.data_test is None:
                print('Please set train data and split ratio or test data.')
            elif self.data_test is None:
                print('Please set test data of split ratio.')
        else:
            if self.data_train is None:
                print('Please set train data.')
            elif self.data_test is None:
                train_index, test_index = train_test_split(self.data_train.index.unique(),
                                                           random_state=42,
                                                           test_size=1-split_ratio)
                self.data_test = self.data_train.loc[test_index]
                self.data_train = self.data_train.loc[train_index]
        cols = np.array(self.data_train.columns)
        size = 0
        for i in cols:
            if str(i).isdigit():
                if int(i) > size:
                    size = int(i)
        if size == 0:
            print("No features columns are found.")
        elif size > 0:
            if self.rubr_id == 'subj':
                y_train = self.data_train.subj
                y_test  = self.data_test.subj
            elif self.rubr_id == 'ipv':
                y_train = self.data_train.ipv
                y_test  = self.data_test.ipv
            elif self.rubr_id == 'rgnti':
                y_train = Codes_helper().cut_rgnti(self.data_train.rgnti)
                y_test  = Codes_helper().cut_rgnti(self.data_test.rgnti)
            X_train = self.data_train[list(map(str, np.arange(size+1)))]
            X_test  = self.data_test[list(map(str, np.arange(size+1)))]
            X_test, y_test = self.__change_test(X_test, y_test)
            y_train = list(y_train)
            return X_train, X_test, y_train, y_test
        return None, None, None, None
    
    # ToDo Description
    def create_w2v_vectors(self, description=None):
        if self.data_train is None:
            print('Please load splitted data to train or train and test fields in Worker object')
            return False
        flag_train, name_train = self.__create_w2v_vectors_on_one_set('train', description=description)
        if self.data_test is not None: 
            flag_test, name_test = self.__create_w2v_vectors_on_one_set('test', description=description)
            return flag_test
        return flag_train
    
    # ToDo Description
    def __create_w2v_vectors_on_one_set(self, data_t='train', description=None):
        """
        Creates pd.DataFrame with vectors instead of text column.

        Args:
        data_t (str): 'test' or 'train'. Data is taken from self.data_x.
        """
        self.__check_conv_type()
        self.__check_w2v()
        if data_t == 'train':
            data = self.data_train
        elif data_t == 'test':
            data = self.data_test
            if description is not None:
                description = 'test'+'_'+description
            else:
                description = 'test'
        else:
            print('data_t can be only "train" or "test".')
            return False, '-'
        columns = list(data.columns)
        columns.remove('text')
        result = pd.DataFrame(columns=columns+list(range(self.w2v_size)))
        total_am = data.index.drop_duplicates().shape[0]
        for j,i in enumerate(data.index.unique()):
            if j%100 == 0: 
                clear_output()
                display(self.conv_type+' '+str(self.w2v_size))
                display(str(j)+'/'+str(total_am))
            if type(data.loc[i]) != pd.core.series.Series:
                features = self.__vectorize(data.loc[i].text.values[0])
                for k in data[columns].loc[i].values:
                    inp = pd.DataFrame([list(list(k) + list(features))], 
                                        columns=columns+list(range(self.w2v_size)), index = [i])
            else:
                features = self.__vectorize(data.loc[i].text)
                inp = pd.DataFrame([list(data.loc[i][columns]) + list(features)], 
                                   columns=columns+list(range(self.w2v_size)), index = [i])
            result = result.append(inp)
        name = self.__create_name("w2v_vectors", result, description=description)
        self.__save_file(name, result)
        if data_t == 'train':
            self.data_train = result
            self.name_train = name
        elif data_t == 'test':
            self.data_test = result
            self.name_test = name       
        return True, name
    
    def create_w2v_model(self, size=50, lang=None, description=None):
        """
        Creates word2vec model based on data_train set. Set new model as current w2v model.

        Args:
        size(int): w2v vectors dimension size.
        """
        self.__check_lang()
        if 'text' in list(self.data_train.columns):
            self.w2v_size = size
            if lang:
                self.lang = lang
            df = pd.concat([self.data_train, self.data_test], ignore_index=True)
            df.text.to_csv('./only_text.csv', index=False, encoding='utf-8')
            model = Word2Vec(LineSentence('./only_text.csv'), size=size, window=4, min_count=3, workers=3)
            os.remove('./only_text.csv')
            self.w2v_model = model
            name = self.__create_name("w2v_model", model, description=description)
            self.__save_file(name, model)
            return True
        else:
            print('Train DataFrame does not contain "text" column.')
            return False
    
    def create_clf(self, model, X_train, X_test, y_train, y_test, parameters=None, description=None, version=1):
        """
        Creates a classifier based on given train and test data and parameters. 
        Saves clf and created description for it's results.

        Args:
        model       -- sklearn model that shoud be trained.
        X_train     -- train objects set (vectors).
        X_test      -- test objects set (vectors).
        y_train     -- rubrics of X_train.
        y_test      -- real rubrics of X_test.
        parameters  -- dict with keys and corresponding  parameters of the classifier.
        description (str): additional data that should be in file name.
        version     -- int or str with version of file. 1 by default.
        """
        clf = model
        if parameters is not None:
            clf.set_params(**parameters)
        clf.fit(X_train, y_train)
        clf_name = self.__create_name('clf_model', clf, version=version, description=description)
        self.__save_file(clf_name, clf)
        self.clf = clf
        pred = []
        for j in clf.predict_proba(X_test):
            all_prob = pd.Series(j, index=clf.classes_)
            pred.append(list(all_prob.sort_values(ascending=False).index))
        stats = self.count_stats(pred, y_test, amounts=[1, 2, 3, 5, -1])
        name = self.__create_name('clf_model', stats, version=version, description=description, info=1)
        self.__save_file(name, stats)
        return clf, clf_name, stats

    def search_for_clf(self, model, parameters, description=None, jobs=3, 
                       skf_folds=3, version=1, scoring='f1_weighted', oneVsAll=False):
        """
        Searches for a best parameters combination and creates a classifier.

        Args:
        model       -- sklearn model that shoud be trained.
        parameters  -- dict with keys and corresponding lists of parameters that should be tested.
        description (str): additional data that should be in file name.
        jobs        -- amount of threads shoud run in parallel during training.
        skf_folds   -- amount of cross-validation folds.
        version     -- int or str with version of file. 1 by default.
        """
        timer = time.time()
        X_train, X_test, y_train, y_test = self.create_sets()
        self.__check_conv_type()
        self.__check_lang()
        skf = StratifiedKFold(
            # TODO: wtf
            # y_train,
            shuffle=True,
            n_splits=skf_folds)
        p = parameters.copy()
        if oneVsAll:
            for i in list(p.keys()):
                p['estimator__'+i] = p.pop(i)
            model = OneVsRestClassifier(model)
        gs = False
        not_gs_parameters = {}
        for i in p.keys():
            if len(p[i]) > 1:
                gs = True
                gs_clf = GridSearchCV(estimator=model,
                                      param_grid=p,
                                      n_jobs=jobs,
                                      scoring=scoring,
                                      cv=skf,
                                      verbose=20)
                gs_clf.fit(X_train, y_train)
                best_parameters = gs_clf.best_estimator_.get_params()
                break
            else:
                not_gs_parameters[i] = p[i][0]
        if gs == False:
            best_parameters = not_gs_parameters
        clf, clf_name, stats = self.create_clf(model, 
                                        X_train, 
                                        X_test, 
                                        y_train, 
                                        y_test,
                                        parameters=best_parameters, 
                                        description=description, 
                                        version=version)
        now = datetime.datetime.today()
        descr = 'Date of creation: ' + str(now.day)+'.'+str(now.month)+'.'+str(now.year)
        if '('in str(self.clf):
            descr += '\nType of classifier:\t' + str(self.clf).split('(')[0]
        else: 
            descr += '\nType of classifier:\t' + str(type(self.clf))
        descr += '\nTested parameters:'
        for i in parameters.items():
            descr += '\n\t' + str(i)[1:-1]
        descr += '\nBest prameters:'
        for i in best_parameters.items():
            descr += '\n\t' + str(i)[1:-1]
        descr += '\nTrain and test data sizes and files:\n' + \
            '\t' + str(len(y_train)) + '\t' + self.name_train + '\n' + \
            '\t' + str(len(y_test)) + '\t' + self.name_test + \
            '\nClassifier version: v' + str(version) 
        if description:
            descr += '\nClassifier remarks:\t' + description
        t = str((time.time() - timer) // 3600) + ' hours\t' +\
            str(((time.time() - timer) % 3600) // 60) + ' minutes\t' +\
            '%.2f'%((time.time() - timer) % 60) + ' seconds'
        descr += '\nTotal training time:\t' + t
        descr += '\nResults (accuracy, precision, recall, f1-score):'
        keys = list(stats.keys())
        keys.sort()
        print('Work time is', t)
        for i in keys:
            mac = stats[i].loc['macro']
            mic = stats[i].loc['micro']
            macro = str(mac['accuracy'].round(5)) + '\t' + str(mac['precision'].round(5)) + '\t' + \
            str(mac['recall'].round(5)) + '\t' + str(mac['f1-score'].round(5))
            micro = str(mic['accuracy'].round(5)) + '\t' + str(mic['precision'].round(5)) + '\t' + \
            str(mic['recall'].round(5)) + '\t' + str(mic['f1-score'].round(5))
            descr += '\n\t\tFor ' + str(i) + ' answers :' + '\n\t Macro ' + macro + '\n\t Micro ' + micro
            print('For '+str(i)+'\n\tmicro '+micro+'\n\tmacro'+macro+'\n')
        name = self.__create_name('clf_model', descr, version=version, description=description, info=1)
        self.__save_file(name, descr)

    def __make_res_b(self, predicts, y_test):
        """
        Counts binary accuracy, precision, recall and f1.

        Args:
        predicts    -- classifiers answers for X_test (0 and 1 for a particular rubric).
        y_test      -- real rubrics of X_test (0 and 1 for a particular rubric).
        """
        ac = accuracy_score(y_test, predicts)
        pr = precision_score(y_test, predicts)
        rec = recall_score(y_test, predicts)
        f1 = f1_score(y_test, predicts)
        return [ac, pr, rec, f1]

    def count_stats(self, predicts, y_test, legend=None, amounts=[1]):
        """
        Counts statistics for predictions of a classifier

        Args:
        predicts    -- classifiers answers for X_test.
        y_test      -- real rubrics of X_test.
        legend      -- list with ordered unique rubrics. If equals to None, legend will be created in alphabet order.
        amounts     -- list with amounts of answers we want to test (-1 means all answers). 1 by default.
        version     -- int or str with version of file. 1 by default.
        """
        if legend is None:
            if self.rubr_id == 'subj':
                legend = Codes_helper(clear_math=not(self.math)).get_codes('subj')
            else:
                legend = [item for sublist in y_test for item in sublist]
                legend = pd.Series(map(str, legend))
                legend = legend.unique()
                legend.sort()
                legend = list(legend)
        if not -1 in amounts:
            amounts = list(map(int,amounts))
            amounts.sort()
            amounts = amounts[::-1]
        else:
            amounts = list(set(amounts)-set([-1]))
            amounts.sort()
            amounts = [-1]+amounts[::-1]
        keys, values = [], []
        for a in amounts:
            k = []
            if a != -1:
                for j in predicts:
                    k += [j[:a]]
            else:
                k = predicts
            cur_pred = k
            stats = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1-score','TP','FP','FN','TN'])
            for i in legend:
                cur_predicts = []
                cur_y_test = []
                for j in zip(cur_pred, y_test):
                    if (type(j[0])==list and i in j[0]) or i==j[0]:
                        cur_predicts += [1]
                    else:
                        cur_predicts += [0]
                    if (type(j[1])==list and i in j[1]) or i==j[1]:
                        cur_y_test += [1]
                    else:
                        cur_y_test += [0]
                temp = []
                for l in self.__make_res_b(cur_predicts, cur_y_test):
                    temp += [l]
                mat = confusion_matrix(cur_predicts, cur_y_test)
                if mat.shape == (1, 1):
                    conf_matr = [0,0,0]+list(np.array(mat).ravel())
                else:
                    conf_matr = list(np.array(mat).ravel())[::-1]
                stats = stats.append(pd.DataFrame([temp+conf_matr],
                                                  columns=['accuracy', 'precision', 'recall', 
                                                           'f1-score', 'TP','FP','FN','TN'], index=[i]))
            stats = stats.sort_index()
            stats_mean = stats.mean().values
            tp, fp, fn, tn = stats_mean[4:]
            acc_temp = (tp + tn) / (tp + fp + fn + tn)
            pr_temp = tp / (tp + fp)
            rec_temp = tp / (tp + fn)
            f1_temp = 2 * pr_temp * rec_temp / (pr_temp + rec_temp)
            stats = stats.append(pd.DataFrame([list(stats_mean[0:4])+['-']*4],
                          columns=['accuracy', 'precision', 'recall', 'f1-score','TP','FP','FN','TN'], 
                                              index = ['macro']))
            stats = stats.append(pd.DataFrame([[acc_temp, pr_temp, rec_temp, f1_temp] +list(stats_mean[4:])],
                          columns=['accuracy', 'precision', 'recall', 'f1-score','TP','FP','FN','TN'], 
                                              index = ['micro']))
            if a != -1:
                keys += [str(a)]
            else:
                keys += ['all']
            values += [stats]
        full_stats = dict(zip(keys, values))
        return full_stats
    
    def __split_all_sect(self, data):
        """
        Splits all rubrics separated with / to different strings.

        Args:
        data (pd.DataFrame): DataFrame that should be splitted.
        """
        timer = time.time()
        am = data.shape[0]
        df = pd.DataFrame(columns=data.columns)
        col = list(data.columns)
        if 'text' in col:
            col.remove('text')
        size = 0
        for i in col:
            if str(i).isdigit():
                if int(i) > size:
                    size = int(i)
        if size != 0:
            for i in list(map(str,np.arange(size+1))):
                col.remove(i)
        for j,i in enumerate(data.index):
            if j%1000 == 0:
                clear_output()
                display('Splitting rubrics '+str(j)+'/'+str(am))
            temp = []
            no_miss = True
            if size == 0:
                text = data.loc[i]['text']
                for l in data.columns:
                    if not l == 'text':
                        if type(data.loc[i][l]) == str:
                            temp.append(str(data.loc[i][l]).split('\\'))
                        else:
                            no_miss = False
                if no_miss:
                    for k in zip_longest(*temp):
                        df = df.append(pd.DataFrame([list(k)+[text]], columns=col+['text'], index=[i]))
            else:
                vect = data.loc[i][list(map(str,np.arange(size+1)))]
                for l in data.columns:
                    if not str(l).isdigit():
                        if type(data.loc[i][l]) == str:
                            temp.append(str(data.loc[i][l]).split('\\'))
                        else:
                            no_miss = False
                if no_miss:
                    for k in zip_longest(*temp):
                        df = df.append(pd.DataFrame([list(k)+list(vect)], 
                                                    columns=col+list(map(str,np.arange(size+1))), index=[i]))
        print('Work time is', int((time.time() - timer) // 3600), 'hours',
              int(((time.time() - timer) % 3600) // 60), 'minutes',
              '%.2f'%((time.time() - timer)%60), 'seconds')
        return df 
    
    def __change_test(self, X_test, y_test):
        """
        Changes test set to make it possible to deal with several answers to one text.

        Args:
        X_test      -- DataFrame with objects features.
        y_test      -- Series with answers corresponding to X_test. 
        """
        df = pd.DataFrame([], columns=X_test.columns)
        ans = []
        for i in X_test.index.unique():
            if type(y_test.loc[i]) == pd.core.series.Series:
                df = df.append(X_test.loc[i].iloc[0])
                ans.append(list(y_test[i][y_test[i].notnull()]))
            else:
                df = df.append(X_test.loc[i])
                ans.append([y_test[i]])
        return df, pd.Series(ans)
    
    ########################################################
    # ToDo: write, test, description
    """
    Unfinished!
    """
    def test_with_new_data(self, data_path):
        self.__check_res_folder()
        self.__check_clf()
        self.__check_w2v()
        self.__check_rubr_id()
        if os.path.exists(data_path):
            self.data_train = pd.read_csv(data_path, index_col=0)
            self.data_train = self.w2v_vectors_creation(self.data_train)
            self.data_test  = self.data_train
            X_train, X_test, y_train, y_test = self.create_sets()
            pred = []
            for j in self.clf.predict_proba(X_test):
                all_prob = pd.Series(j, index=self.clf.classes_)
                pred.append(list(all_prob.sort_values(ascending=False).index))
            stats = self.count_stats(pred, y_test, amounts=[1,2,3,5,-1])
            name = self.__create_name('result', stats, description='test', info=1)
            self.__save_file(name, stats)
            for i in stats.keys():
                mac = stats[i].loc['macro']
                mic = stats[i].loc['micro']
                macro = str(mac['accuracy'].round(3)) + '\t' + str(mac['precision'].round(3)) + '\t' + \
                str(mac['recall'].round(3)) + '\t' + str(mac['f1-score'].round(3))
                micro = str(mic['accuracy'].round(3)) + '\t' + str(mic['precision'].round(3)) + '\t' + \
                str(mic['recall'].round(3)) + '\t' + str(mic['f1-score'].round(3))
                print('For '+str(i)+'\n\tmicro '+micro+'\n\tmacro '+macro+'\n')
            answers = []
            all_prob = pd.Series(j, index=self.clf.classes_)
            res = all_prob.sort_values(ascending=False)
            res = res[res!=0]
            temp = ''
            for i, k in zip(res, res.index):
                temp += k+'-'+str(i).replace('.',',')+'\\'
            answers.append(temp[:-1])
            pred = pd.DataFrame(list(zip([self.rubr_id]*len(answers), answers, 
                             ['###']*len(answers))), columns=['rubric id','result', 'correct'], index=X_train.index)
            name = self.__create_name('answers', pred, description='test', info=1)
            self.__save_file(name, pred)
        else:
            print('Please specify existing test data path.')
            return False
        
    ########################################################
    
    def __save_file(self, name, save_data, check_path=False):
        """
        Saves the data with the given file name.

        Args:
        name (str): saving file name.
        file_type   -- string/DataFrame/dict{int or str:pd.DataFrame}/w2v model/clf model for saving.
        """
        path = os.path.split(name)[0]
        if check_path:
            while not os.path.exists(path):
                print(name)
                print('Not a valid path. Please enter full name:')
                name = input()
                path = os.path.split(name)[0]
        if os.path.exists(path):
            if   type(save_data) == dict:
                writer = pd.ExcelWriter(name, engine='xlsxwriter')
                names = list(map(str, save_data.keys()))
                names.sort()
                for i in names:
                    save_data[i].to_excel(writer, sheet_name=self.rubr_id+'_'+str(i))
                writer.save()
            elif type(save_data)   == str:
                f = open(name, 'w')
                f.write(save_data)
                f.close()
            elif type(save_data) == pd.core.frame.DataFrame:
                save_data.to_csv(name, encoding='utf-8', sep='\t')
            elif type(save_data) == Word2Vec:
                save_data.save(name)
            else:
                joblib.dump(save_data, name)
            return True
        else:
            print('Something went wrong.')
            return False
    
    # check_w2v_model()
    def __vectorize(self, text): 
        """
        Transforms one text into vector.

        Args:
        text (str): string with text taht should be transformed.
        """
        self.__check_conv_type()
        if self.w2v_model:
            tokens = text.split()
            features = [0]*self.w2v_size
            if self.conv_type == 'sum':
                for t in tokens:
                    if t in self.w2v_model:
                        features += self.w2v_model[t]
            elif self.conv_type in ['max','mean']:
                for t in tokens:
                    if t in self.w2v_model:
                        features = np.vstack((features, self.w2v_model[t]))
                if features.shape[0] > 1:
                    if self.conv_type == 'max':
                        features = features.max(axis=0)
                    else:
                        features = features.mean(axis=0)
            return features
        else:
            print('Word2Vec model is not set.')
            return None
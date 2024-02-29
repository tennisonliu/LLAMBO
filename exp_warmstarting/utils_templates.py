from abc import ABC, abstractmethod
import inspect
import json
import os

current_file = inspect.getfile(inspect.currentframe())
CURRENT_DIR = os.path.dirname(os.path.abspath(current_file))

import ConfigSpace as CS

def load_json(path_json):
    f = open(path_json)
    data = json.load(f) 
    f.close()
    return data

def save_json(path_json, data):
    # Write the JSON data to the file
    # Convert numpy.int64 values to int
    for item in data:
        for key, value in item.items():
            if isinstance(value, np.int64):
                item[key] = int(value)

    with open(path_json, "w") as json_file:
        json.dump(data, json_file, indent=4)  

def hyperparameter_context(config_space, add_ranges = False):
    str_result = ''
    for hp in config_space.get_hyperparameters():
        str_result += hp.name
        if isinstance(hp, CS.CategoricalHyperparameter):
            choices = hp.choices
            if choices[0] == False and choices == 'True':
                str_result  += ' (boolean), ' if not add_ranges else ' (boolean between {false, true}), ' 
            else:
                str_result += ' (string), ' if not add_ranges else  ' (string %s), ' % str(choices).replace('[', '{').replace(']', '}')
        else:
            lower, upper =  hp.lower, hp.upper
            if isinstance(hp, CS.UniformIntegerHyperparameter):
                str_result += ' (integer), ' if not add_ranges else f' (integer between [{lower}, {upper}]), ' 
            elif isinstance(hp, CS.UniformFloatHyperparameter):
                str_result += ' (float), ' if not add_ranges else f' (float between [{lower}, {upper}]), '
    str_result = str_result[:-2]
    return str_result

# Reading text from a file
def read_text_from_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            content = content.replace('\n\n', '[DOBLE]')
            content = content.replace('\n', ' ')
            content = content.replace('[DOBLE]', '\n')
            return content
    except Exception as e:
        print(f"Error reading from '{filename}': {e}")
        return None

def add_context_information(text, config_space, num_recommendation = 10, text_dict = None, provide_ranges = False):
    if text_dict is not None:
        for key in text_dict.keys():
            text = text.replace(key, text_dict[key])
    text     = text.replace('[NUM_RECOMMENDATION]', str(num_recommendation))
    hyp_text = hyperparameter_context(config_space, provide_ranges)
    text     = text.replace('[CONFIGURATION_AND_TYPE]', hyp_text)
    return text

def obtain_path_to_write_config(task_name, context_type = None):
        import os
        path_results  = "init-configs"        
        if not os.path.exists(path_results):
                os.makedirs(path_results)
        path_task     = '%s/%s' % (path_results, task_name)
        if not os.path.exists(path_task):
                os.makedirs(path_task)
        if context_type is not None:
            path_context  = '%s/%s' % (path_task, context_type) 
            if not os.path.exists(path_context):
                    os.makedirs(path_context)
            return path_context
        else:
            return path_task
# Abstract class
class TemplateReader(ABC):
    def __init__(self, path):
        self.path_to_read = path

    @abstractmethod
    def obtain_dict_text(self):
        pass

    def read_text(self):
        return read_text_from_file(self.path_to_read)

    def add_context(self, config_space, num_recommendation = 10, task_dict = None):
        text              = read_text_from_file(self.path_to_read)
        text_dict         = self.obtain_dict_text(task_dict)
        text_with_context = add_context_information(text, config_space, num_recommendation = num_recommendation,
                                                    text_dict = text_dict, provide_ranges = self.provide_ranges)
        return text_with_context

    def write_all_config(self, this_path, all_config):
        for idx, config in enumerate(all_config):
            save_json('%s/config%s.json' % (this_path, idx), config)

    def read_all_config(self, this_path, num_configs = 10):
        all_config = []
        idx = 0
        while True:
            if len(all_config) == num_configs:
                break
            this_load = load_json('%s/config%s.json' % (this_path, idx))
            if len(this_load) >= 5:
                all_config += [this_load]
            idx += 1

        return all_config

# Subclasses
ALL_TEMPLATES_PATH = {
    'Full_Context':    f'{CURRENT_DIR}/templates/classification_fullcontext.txt',
    'No_Context':      f'{CURRENT_DIR}/templates/classification_nocontext.txt',
    'Partial_Context': f'{CURRENT_DIR}/templates/classification_partialcontext.txt',
}

class FullTemplate(TemplateReader):
    def __init__(self, context = '', provide_ranges = True, add_name = None):
        self.context_template = context
        #self.add_path = 
        super().__init__(ALL_TEMPLATES_PATH[self.context_template])
        self.provide_ranges  = provide_ranges
        self.add_name = add_name
        self.add_path = self.obtain_add_path()

    def obtain_add_path(self):
        if self.provide_ranges:
            add_path = "Range_" + self.context_template
        else:
            add_path = "NoRange_" + self.context_template
        if self.add_name is not None:
            add_path += '-%s' % self.add_name
        return add_path

    def obtain_dict_text(self, task_dict):
        new_dict = {}
        new_dict['[MODEL]']            = str(task_dict['model'])
        new_dict['[TASK]']             = str(task_dict['task'])
        new_dict['[METRIC]']           = str(task_dict['metric'])
        new_dict['[NUM_SAMPLES]']      = str(task_dict['num_samples'])
        new_dict['[NUM_FEATURES]']     = str(task_dict['num_feat'])        
        new_dict['[NUM_NUM_FEATURES]'] = str(task_dict['num_feat_cont'])
        new_dict['[NUM_CAT_FEATURES]'] = str(task_dict['num_feat_cat'])
        new_dict['[NUM_BY_CLASS]']     = self.obtain_str_class(task_dict['num_by_class'])
        if task_dict['task'] == 'classification':
            new_dict['[ADD_CLASS_INFO]'] = 'Class distribution is %s.' % new_dict['[NUM_BY_CLASS]']
        else:
            new_dict['[ADD_CLASS_INFO]'] = ''
        if self.context_template == 'Extended_Context':
            new_dict['[NUM_STAT]']         = self.add_str_num(task_dict['num_feat_cont'], str(task_dict['skew'])[1:-1], str(task_dict['kurtosis'])[1:-1] )
            new_dict['[CORR_FEATURES]']    = str(task_dict['num_x_corr_pass'])
            new_dict['[CORR_TARGET]']      = str(task_dict['num_y_corr_pass'])
            new_dict['[NUM_POSSIBLE]']     = str(task_dict['num_x_corr'])
        return new_dict

    def add_str_num(self, num_feat, skew, kurtosis):
        if num_feat > 0:
            this_str = 'We are standarizing numerical values to have mean 0 and std 1. The Skewness of each feature is %s and the kurtosis is %s.' % (skew, kurtosis)  
        else:
            this_str = ''
        return this_str 

    def obtain_str_class(self, num_by_class):
        this_str  = ''
        num_class = len(num_by_class)
        for idx, this_class in enumerate(num_by_class):
            if 0 < idx < num_class - 1:
                this_str += ', '
            if idx == num_class - 1:
                this_str += ' and '

            this_str += str(this_class)
            this_str += ' (y=%s)' % idx                
        return this_str 

    def obtain_path(self, path_name, provide_ranges = False):
        add_path = self.add_path
        return obtain_path_to_write_config(path_name, add_path)

class RandomTemplate(TemplateReader):
    def __init__(self, name = None, add_name = None):
        super().__init__(f'{CURRENT_DIR}/templates/random.txt')
        self.name = name
        self.add_name = add_name

    def obtain_dict_text(self, task_dict = None):
        return None

    def obtain_path(self, path_name, provide_ranges = False):
        if self.name is None:
            add_path = "Random"
        else:
            add_path = "Random-%s" % self.name
        if self.add_name is not None:
            add_path += '-%s' % self.add_name
        return obtain_path_to_write_config(path_name, add_path)

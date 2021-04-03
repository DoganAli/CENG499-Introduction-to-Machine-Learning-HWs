from math import log2

def divide(data, attr_index, attr_vals_list):
    """Divides the data into buckets according to the selected attr_index.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: A list that includes K data lists in it where K is the number
     of values that the attribute with attr_index can take
    """
    
    bucket_list = []
    for val in attr_vals_list[attr_index]:
        data_val = [d for d in data if d[attr_index] == val]
        bucket_list.append(data_val)
        
    return bucket_list


def entropy(data, attr_vals_list):
    """
    Calculates the entropy in the current data.
    :param data: Current data in the node
    :param attr_vals_list: List of values that attributes may take
    (Last attribute is for the labels)
    :return: Calculated entropy (float)
    """
    if(len(data) == 0 ):
        return 0 
    labels = attr_vals_list[-1]
    entropy = 0 
    n = len(data)
    frequency_dict = {}
    for l in labels:
        frequency_dict[l] = 0 
    for d in data :
        label = d[-1]
        frequency_dict[label] +=1
    
    for label in labels :
        p = frequency_dict[label] / n 
        if( p > 0 ):
            entropy += - p * log2(p)
           
    return entropy


def info_gain(data, attr_index, attr_vals_list):
    """
    Calculates the information gain on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: information gain (float), buckets (the list returned from divide)
    """
    n = len(data)
    buckets = divide(data,attr_index,attr_vals_list)
    data_entropy = entropy(data,attr_vals_list)
    average_entropy = 0.0 
    
    for data_subset in buckets :
        average_entropy += ( len(data_subset)/ n )*entropy(data_subset,attr_vals_list)
    
    information_gain = data_entropy - average_entropy
    return information_gain,buckets


def gain_ratio(data, attr_index, attr_vals_list):
    """
    Calculates the gain ratio on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: gain_ratio (float), buckets (the list returned from divide)
    """
    if(len(data) == 0): 
        return 0 
    
    gain,buckets = info_gain(data, attr_index, attr_vals_list)
    int_i = 0
    freq_intI = {}
    n = len(data)
    
    for attr in attr_vals_list[attr_index]:
        freq_intI[attr] = 0 
        
    for d in data :
        attr = d[attr_index]
        freq_intI[attr] += 1
    
    for attr in attr_vals_list[attr_index]:
        p = freq_intI[attr] / n 
        if(p > 0):
            int_i += - p * log2(p)
        
    return gain/int_i , buckets


def gini(data, attr_vals_list):
    """
    Calculates the gini index in the current data.
    :param data: Current data in the node
    :param attr_vals_list: List of values that attributes may take
    (Last attribute is for the labels)
    :return: Calculated gini index (float)
    """
    gini_index = 1 
    buckets = divide(data,-1,attr_vals_list)
    n = len (data)
    for b in buckets:
        p = len(b) / n
        gini_index -= p*p 
    
    return gini_index


def avg_gini_index(data, attr_index, attr_vals_list):
    """
    Calculates the average gini index on the current data when the attribute with attr_index is selected.
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: average gini index (float), buckets (the list returned from divide)
    """
    
    buckets = divide(data, attr_index, attr_vals_list)
    
    avg_gini = 0
    n = len(data)
    for b in buckets :
        p = len(b) / n 
        if(p > 0) :
            avg_gini += p * gini(b,attr_vals_list)
        
        
    return avg_gini,buckets


def chi_squared_test(data, attr_index, attr_vals_list):
    """
    Calculated chi squared and degree of freedom between the selected attribute and the class attribute
    :param data: Current data in the node
    :param attr_index: Selected attribute index to partition the data
    :param attr_vals_list: List of values that attributes may take
    :return: chi squared value (float), degree of freedom (int)
    """
    #create an obs table where rows are class labels, columns are attribute values
    buckets_class = divide(data,-1,attr_vals_list)
    obs_data = []
    exp_table = []
    obs_table = []
    n = len(data)
    for b in buckets_class:
        obs_constant_class = divide(b,attr_index,attr_vals_list)
        obs_data.append(obs_constant_class)
        
    for row in obs_data :
        obs_row = []
        exp_row = []
        for col in row :
            obs_row.append(len(col))
            exp_row.append(0)
        obs_table.append(obs_row)
        exp_table.append(exp_row)
    
    ni = []
    nj = [0 for d in obs_table[0]]
    for i in range(len(obs_table)):
        ni.append( sum(obs_table[i]))
        for j in range(len(obs_table[0])):
            nj[j] += obs_table[i][j]
    
    for i in range(len(ni)):
        for j in range(len(nj)):
            exp_table[i][j] = ni[i]*nj[j] / n
            
    chi_value = 0     
    for i in range(len(obs_table)):
        for j in range(len(obs_table[0])):
            if(exp_table[i][j] != 0 ):
                chi_value +=((obs_table[i][j]-exp_table[i][j])**2) / exp_table[i][j]
    

    new_ni = [1 for i in ni if i != 0 ]
    new_nj = [1 for j in nj if j != 0 ]
    degree_freedom = (sum(new_ni)-1)*(sum(new_nj)-1)
    
    return chi_value , degree_freedom








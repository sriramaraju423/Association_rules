#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


transactions_retail = []
with open(r"C:\Users\srira\Desktop\Ram\Data science\Course - Assignments\Module 15 - Association rules\Dataset\transactions_retail1.csv") as f:
    transactions_retail = f.read()
transactions_retail


# In[ ]:


transactions_retail = transactions_retail.replace("'","")
transactions_retail


# In[ ]:


transactions_retail = transactions_retail.replace('"',"")
transactions_retail


# In[ ]:


transactions_retail = transactions_retail.replace('&',"")
transactions_retail


# In[ ]:


transactions_retail = transactions_retail.split("\n")
transactions_retail


# In[ ]:


transactions_retail_lst = []
for i in transactions_retail:
    transactions_retail_lst.append(i.split(","))
transactions_retail_lst


# In[ ]:


#Using recursion concept in loops to implement this logic


# In[ ]:


def naremoval(list):
    if 'NA' not in list:
        return list
    list.remove('NA')
    return naremoval(list)


# In[ ]:


for i in transactions_retail_lst:
    naremoval(i)
transactions_retail_lst


# In[ ]:


transactions_retail_count = []
type(transactions_retail_count)


# In[ ]:


#Let's generate the matrix format transactions which is needed for association rules


# In[ ]:


#Converting to dataframe


# In[ ]:


transactions_retail_df = pd.DataFrame(pd.Series(transactions_retail_lst))
transactions_retail_df.head(10)


# In[ ]:


transactions_retail_df.columns = ['transactions']
transactions_retail_df


# In[ ]:


Binary_transactions_retail = transactions_retail_df['transactions'][:7042].str.join(sep="*").str.get_dummies(sep="*")
Binary_transactions_retail


# In[ ]:


for cat in Binary_transactions_retail.columns:
    transactions_retail_count.append(Binary_transactions_retail[cat].value_counts()[1])
transactions_retail_count


# In[ ]:


type(transactions_retail_count[0])


# In[ ]:


transactions_retail_count.sort(reverse = True)
transactions_retail_count


# In[ ]:


#Tried sorting using some other way but not so great let's continue lamda appraoch


# In[ ]:


transactions_retail_lst = transactions_retail_lst[:7042]
all_retail_list = [i for item in transactions_retail_lst for i in item]
all_retail_list


# In[ ]:


len(all_retail_list)


# In[ ]:


from collections import Counter


# In[ ]:


retail_items_freq = Counter(all_retail_list)
retail_items_freq


# In[ ]:


retail_items_freq = sorted(retail_items_freq.items(),key=lambda x:x[1])
retail_items_freq


# In[ ]:


items = list(reversed([i[0] for i in retail_items_freq]))
frequencies = list(reversed([i[1] for i in retail_items_freq]))
items,frequencies


# In[ ]:


#Plotting the top 10 selled products


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.bar(height=frequencies[0:10], x=items[0:10], color="rgby");plt.xticks(items[0:10]);plt.xlabel('Retail Products');plt.ylabel('count')
plt.show()


# In[ ]:


#Let's build rules


# In[ ]:


from mlxtend.frequent_patterns import apriori,association_rules


# In[ ]:


retail_rules = apriori(Binary_transactions_retail, min_support=0.005, max_len=3, use_colnames=True)
retail_rules


# In[ ]:


retail_rules.sort_values('support', ascending=False, inplace=True)
retail_rules


# In[ ]:


#Plotting top rules


# In[ ]:


retail_rules['itemsets_plot'] = retail_rules['itemsets'].apply(lambda x:', '.join(list(x))).astype('unicode')
retail_rules


# In[ ]:


#plotting


# In[ ]:


import numpy as np
plt.figure(figsize=(15,5))
plt.barh(y=range(0,11),width=retail_rules.support[0:11],color="rgby");plt.xlabel('support');plt.ylabel('items/rules')
#annotating the bar plot
x = np.array(retail_rules.support[0:11])
y = range(0,11)
i=0
for x,y in zip(x,y):
    while(i<len(retail_rules.index)):
        plt.annotate(retail_rules.itemsets_plot[retail_rules.index[i]],(x,y),textcoords="offset points",xytext=(0,10))
        i+=1
        break
plt.show()


# In[ ]:


retail_rules_all = association_rules(retail_rules, metric='lift', min_threshold=1)
retail_rules_all


# In[ ]:


retail_rules_all.sort_values('lift', ascending=False, inplace=True)
retail_rules_all


# In[ ]:


#Plotting top 10 rules


# In[ ]:


retail_rules_all['antecedents'] = retail_rules_all['antecedents'].apply(lambda x:', '.join(list(x))).astype('unicode')
retail_rules_all['consequents'] = retail_rules_all['consequents'].apply(lambda x:', '.join(list(x))).astype('unicode')
retail_rules_all


# In[ ]:


#Putting together antecendent & consequent for annotations
ant_cons = []
for i in retail_rules_all.index:
    while(i<len(retail_rules_all.antecedents) and i<len(retail_rules_all.consequents)):
        ant_cons.append("("+retail_rules_all.antecedents[i] +") & ("+ retail_rules_all.consequents[i]+")")
        break
ant_cons[:11]


# In[87]:


plt.figure(figsize=(15,7))
plt.barh(y=range(0,11),width=retail_rules_all.lift[0:11],color="rgby");plt.xlabel('lift');plt.ylabel('retail items');plt.xticks(range(0,11))
x = np.array(retail_rules_all.lift[0:11])
y = range(0,11)
i=0
for x,y in zip(x,y):
    while(i<11):
        plt.annotate(ant_cons[i],(x,y),textcoords="offset points",xytext=(0,10))
        i+=1
        break
plt.show()        


# In[ ]:


Binary_transactions_retail.columns[0:10]


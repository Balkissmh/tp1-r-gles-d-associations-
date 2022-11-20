import numpy as np
import pandas as pd
import itertools

#lecture de dataframe
data = pd.read_table('market_basket.txt')

#affichier 10 premiers ligne
print(data.head(10))
print(data.iloc[1:11])

#taille de dataframe
print(data.shape)

# Question 4
col=data['Product'].sort_values().unique()
lig=data['ID'].unique()

matrix=np.zeros((len(lig),len(col)))
k=0
for i in lig:
  df=data[data["ID"]==i]
  a=0
  for j in col:
    if df['Product'].str.contains(j).any():
      matrix[k][a]=1
    a=a+1
  k=k+1

df = pd.DataFrame(matrix, index=lig,columns=col)
print(df)


# Question 5 
df=pd.crosstab(data['ID'],data['Product'])
print(df)

# Question 6
df.iloc[:3,:30]

# Question 7

nbr_max_produit=2 #nbr max doit etre 4 mais ça sature la mémoire
def ExtractItemset(df, n):
    return list(itertools.combinations(df, n))

def CalculSupport (df,subsets,n):
  tab=[]
  k=0
  for i in subsets:
    subset=df[list(i)]
    w=subset[subset.sum(axis='columns')==n].count()
    tab.append(w[0])
    k=k+1
  return(tab)

# le main du programme 
min_supp=0.025*len(lig)
for i in range(1,nbr_max_produit+1):
  itemset=ExtractItemset(col,i)
  Support=CalculSupport(df,itemset,i)
  ff= pd.DataFrame(itemset)
  sup=pd.DataFrame(Support)
  sup.columns=['Support']# Renommer la colonne sup
  frame=[ff,sup]
  F=pd.concat(frame,axis=1)
  C=F[F["Support"]>min_supp] #filtrer avec la contition min_sup 
  print("C",i)
  print(C)
   
def Regle(item):
  it=item[:-1]
  for i in range(1,nbr_max_produit):
    List=ExtractItemset(it,i)
    PremisList=List[:-i]
    CclList=List[i:]
    PremisSupport=CalculSupport(df,PremisList,i)
    TotalSupport=CalculSupport(df,[it],nbr_max_produit)
  return(PremisList,CclList,np.array(TotalSupport)/np.array(PremisSupport))



AprioriRes=[]
k=0
for i in C.index:
  k=k+1
  ligne=list(C.loc[i,:])
  Premis,Ccl,Confidence=Regle(ligne)
  print('Regle :',Premis,' -> ',Ccl,' With confidence',Confidence)


  from mlxtend.frequent_patterns import apriori, association_rules
frq_items = apriori(df, min_support = 0.025, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

# Question 8
print(rules[:15])


# Question 9
def is_inclus(_subset,_set):
  if len(_subset)>len(_set):
    return(False)
  for i in _subset:
    if not(i in _set):
      return(False)
  return(True)
print(is_inclus(['balkiss','mahfoudh'],['balkiss','mahfoudh','3IDL02']))


# Question 10
Rules_itemset=[list(a)+list(c) for a,c in zip(rules['antecedents'],rules['consequents'])]
for i in Rules_itemset:
  if is_inclus(['Aspirin'],i):
    print(i)
    
    
 #Question 11
for i in Rules_itemset:
  if is_inclus(['Aspirin','Eggs'],i):
    print(i)   
    
    
#Question 12
rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.75)

# Question 13
print(rules[:5])

# Question 14
print(rules[rules.lift>7])

# Question15
print(rules[rules.consequents.str.contains('2pct_Milk', na=True)] )

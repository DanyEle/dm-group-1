#Parameters
#df: dataFrame
#mink: min value of k for k-means 
#maxk: max value of k for k-means
#numberOfColumn: number of columns to involve in each combinations of attribute (ES. if numberOfColumn=3, we will have a combination like "Age, Status, Limit")
def kmeans_(df, mink, maxk, numberOfColumns):
	#create all combinations of attributes
    L = df.columns
    comb=[",".join(map(str, comb)) for comb in combinations(L, numberOfColumns)]
    d=({})
    sse = list()
    sil= list()
    labels=list()
    centers=list()
    
	#for each combination
    for c in comb:
        print(c)
        x=c.split(",")[0]
        y=c.split(",")[1]
        z=c.split(",")[2]
		#takes columns
        data=df[[x,y,z]]
		#normalize data
        X = scaler.fit_transform(data.values)
		
		#run kmeans with different value of k
        for k in range(mink,maxk):
            kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=100)
            kmeans.fit(X
			#for each k, we store sse,labels,centers, silhouette
            sse.append(kmeans.inertia_)
            labels.append(kmeans.labels_)
            centers.append(kmeans.cluster_centers_)
            sil.append(silhouette_score(X, kmeans.labels_))
        
		#for each combinations of attribute we store a list of silhouette, sse, labels and centers
        d[c]=({'Sil':sil,
               'Sse':sse,
               'Labels':labels,
               'Centers':centers})
			   
		#if you want see the best value of silhouette obtained from a combination c, the instruction are the following:
		#print(max(sil)) 
		#print(sil.index(max(sil))+mink)
	
	#at the end we have a dictionary where the keys are the combinations and for each combinations we have all the informations 

	#we serialize the dictionary 
    import pickle
    pickle.dump(d, open('kmeansProva.p', 'wb'))
	
	#if we want reload the dictionary, the instruction is the following:
	#result=pickle.load(open('kmeansProva.P', 'rb'))
    return d
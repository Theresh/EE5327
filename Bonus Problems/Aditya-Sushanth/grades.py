from sklearn.cluster import KMeans
import numpy as np
with open("marks.csv", "r") as fp:
    marks = fp.read().split("\n")

for i in range(len(marks)):
    marks[i]=np.array([float(marks[i])])

marks = np.array(marks)
print(marks)
kmeans = KMeans(n_clusters=7, random_state=0).fit(marks)


print(kmeans.labels_)
labels = kmeans.labels_

print(kmeans.cluster_centers_)



means = [k[0] for k in kmeans.cluster_centers_]

a=sorted(zip(means,range(7)),key=lambda x:x[0])

grade_dict = {a[0][1]: "D", a[1][1]: "C-", a[2][1]: "C",
              a[3][1]: "B-", a[4][1]: "B", a[5][1]: "A-", a[6][1]: "A"}


with open("grades_results.csv", "w") as fp:
    for i,label in enumerate(labels):
        fp.write("{} {}\n".format(marks[i][0], grade_dict[label]))
        



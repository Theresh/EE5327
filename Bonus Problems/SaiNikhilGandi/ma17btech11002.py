from array import *
import math


roll = ["CH18MTECH11008",	
"CS18RESCH11002",
"CS19RESCH01001",	
"EE15B19M000002",	
"EE15BTECH11003",	
"EE15BTECH11008",	
"EE15BTECH11016",	
"EE15BTECH11029",	
"EE15BTECH11032",
"EE15BTECH11035",	
"EE16BTECH11006",	
"EE16BTECH11022",	
"EE16BTECH11023",	
"EE16BTECH11024",	
"EE16BTECH11031",	
"EE16BTECH11036",	
"EE16BTECH11038",	
"EE16BTECH11040",	
"EE16BTECH11043",	
"EE18ACMTECH11005",	
"EE18ACMTECH11006",	
"EE18MTECH11001",	
"EE18MTECH11002",	
"EE18MTECH11006",	
"EE18MTECH11007",	
"EE18MTECH11015",	
"EE18MTECH11016",	
"EE18MTECH11017",	
"EE18MTECH11026",	
"EE19MTECH01007",	
"EE19MTECH01008",	
"ES15BTECH11009",	
"ES16BTECH11002",	
"ES16BTECH11007",	
"ES16BTECH11021",	
"MA17BTECH11001",	
"MA17BTECH11002",	
"MA17BTECH11003",	
"MA17BTECH11004",	
"MA17BTECH11005",	
"MA17BTECH11006",	
"MA17BTECH11007",	
"MA17BTECH11008",	
"MA17BTECH11009",	
"MA17BTECH11010",	
"EE16BTECH11019",	
"EE16BTECH11011",	
"EE16BTECH11014"]

marks=array('f',[17.2,
0,
0,
76.7,
66.7,
74.2,
10,
10,
95,
26.1,
15,
99.2,
99.2,
0,
0,
0,
0,
96.4,
10,
27.7,
88.4,
86.7,
84.2,
26.9,
91.7,
95.7,
94.2,
96.7,
17.2,
0,
93.2,
74.7,
0,
8,
90.1,
92.5,
59.2,
102,
90,
92.5,
97.5,
50.4,
25.2,
96,
86.5,
72.6,
60.4,
63.4])

marks1=array('f',[17.2,
0,
0,
76.7,
66.7,
74.2,
10,
10,
95,
26.1,
15,
99.2,
99.2,
0,
0,
0,
0,
96.4,
10,
27.7,
88.4,
86.7,
84.2,
26.9,
91.7,
95.7,
94.2,
96.7,
17.2,
0,
93.2,
74.7,
0,
8,
90.1,
92.5,
59.2,
102,
90,
92.5,
97.5,
50.4,
25.2,
96,
86.5,
72.6,
60.4,
63.4])

mean=0

for y in marks:
	mean=mean+y


mean=mean/48

for j in range(0,48):
	marks1[j]=marks1[j]-mean

for j in range(0,48):
	marks1[j]=marks1[j]*marks1[j]

sd=0

for k in marks1:
	sd=sd+k


sd=sd/48
sd=math.sqrt(sd)

ten=mean+1.5*sd
nine=mean+1.0*sd
eight=mean+0.5*sd
seven=mean
six=mean-0.5*sd
five=mean-1.0*sd
four=mean-1.5*sd


print" The mean of scores is "
print(mean)

print"The standard deviation of the scores is "
print(sd)

print"\n"


for z in range(0,48):


	if marks[z]>ten:
		print"The grade of " +roll[z]+" is A"

	elif marks[z]>nine and marks[z]<ten:
		print"The grade of " +roll[z]+" is A"


	elif marks[z]>eight and marks[z]<nine:
		print"The grade of " +roll[z]+" is A-"


	elif marks[z]>seven and marks[z]<eight:
		print"The grade of " +roll[z]+" is B"


	elif marks[z]>six and marks[z]<seven:
		print"The grade of " +roll[z]+" is B-"


	elif marks[z]>five and marks[z]<six:
		print"The grade of " +roll[z]+" is C"


	elif marks[z]>four and marks[z]<five:
		print"The grade of " +roll[z]+" is C-"


	elif marks[z]<four:
		print"The grade of " +roll[z]+" is D"



		

#Thi contains program from grader script.
#Given the list of marks each in a new line in a file named "marks.txt", it reads and outputs a file 
#Output file "res.txt" contains marks followed by grades based on Relative grading
#Simply copy the column of marks from the sheet and paste in input file.
#To get only grades, just modify the 60 th line(remove 'str(marsk[i])+"\t"') and paste the contents of outputfile in a new column
from array import *
import math
infile = open("marks.txt","r")
total=0
count=0
marks=[]
marks_copy=[]
grades=[]
for line in infile:
	marks.append(float(line))
	marks_copy.append(marks[count])
	total = total + marks[count]
	count=count+1
print("Mean is "+str(total/count))
total2=0
for i in marks_copy:
	i = i-(total/count)
	i=i*i
	total2=total2+i
total2=total2/count
total2=math.sqrt(total2)

mean = total/count
print("Standard deviation is :"+str(total2))
ten = mean + 1.5* total2
nine = mean + 1.0* total2
eight = mean + 0.5* total2
seven = mean 
six = mean - 0.5* total2
five = mean - 1.0* total2
four = mean - 1.5* total2
gradelist=[]
for num in marks:
	if num >ten :
		gradelist.append("A+")
	if num<ten and num>nine:
		gradelist.append("A")
	if num<nine and num>eight:
		gradelist.append("A-")
	if num<eight and num>seven:
		gradelist.append("B")
	if num<seven and num>six:
		gradelist.append("B-")
	if num<six and num>five:
		gradelist.append("C")
	if num<five and num>four:
		gradelist.append("C-")
	if num<four:
		gradelist.append("D")

outfile = open("res.txt","w")
j=0
outfile.write("Marks\tGrades\n")
for i in gradelist:
	string = str(marks[j])+"\t"+str(i)+"\n"
	j=j+1
	outfile.write(string)
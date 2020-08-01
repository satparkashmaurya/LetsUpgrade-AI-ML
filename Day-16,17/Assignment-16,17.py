import pandas as pd
import matplotlib as plt

titles =pd.read_csv('G:/AIML/Assignments-Received/Day-16-17/titles.csv')

titles.head()
titles.describe
titles.index
titles.shape


# Q1. How many movies are listed in the titles dataframe?
titles.count(axis=0)
print("Number of Movies:  :",len(titles.axes[0])) #The len() function returns the number of items in an object. When the object is a string, the len() function returns the number of characters in the string.

# Q2. What are the earliest two films listed in the titles dataframe?
titles.nsmallest(2,'year')

titles.set_index(['title','year'])

titles.head()
titles.shape
titles.describe
titles.index


# Q3. How many movies have the title "Hamlet"?
titles = titles.set_index('title')
result = titles.loc['Hamlet']
print(result)
nf=titles.loc[['Hamlet']] #Access a group of rows and columns by label(s) or a boolean array.
print("Number of Movies with Title 'Hamlet' are :",nf.size)


# Q4. How many movies have the title "North by Northwest"?
nf2=titles.loc[['North by Northwest']]
print("Number of Movies with Title 'North by Northwest' are :",nf2.size)


# Q5. When was the first movie titled "Hamlet" made?
nf.nsmallest(1,'year')


# Q6. List all of the "Treasure Island" movies from earliest to most recent
nf3=titles.loc[['Treasure Island']]
nf4 = nf3.sort_values(by=['year'])
print(nf4)


# Q7. How many movies were made in the year 1950?
nf5=titles.loc[titles['year'] == 1950]
print("Number of Movies made in year 1950 are :",nf5.size)


# Q8. How many movies were made in the year 1960
nf6=titles.loc[titles['year'] == 1960]
print("Number of Movies made in year 1960 are :",nf6.size)


# Q9. How many movies were made from 1950 through 1959?
i=1950
c=0
while i < 1960 :
    nf7=titles.loc[titles['year'] == i]
    num=nf7.size
    c=c+num
    i=i+1
print("Total Movies from 1950 through 1959 : ", c)


# Q10. In what years has a movie titled "Batman" been released?
nf8=titles.loc[['Batman']]
nf9 = nf8.sort_values(by=['year'])
print(nf9)


# Q11. How many roles were there in the movie "Inception"?
cast=pd.read_csv('G:/AIML/Assignments-Received/Day-16-17/cast.csv')
cast.head()
rd=pd.read_csv('G:/AIML/Assignments-Received/Day-16-17/release_dates.csv')
rd.head()
cast = cast.set_index('title')
resulti = cast.loc['Inception']
print(resulti)
nf10=cast.loc[['Inception']] 
print("Number of Roles in the Title 'Inception' are :",nf10.size)

# Q12. How many roles in the movie "Inception" are NOT ranked by an "n" value?
nf10.shape
print("Roles Not Ranked in Movie Inception : ",nf10['n'].isna().sum())

# Q13. But how many roles in the movie "Inception" did receive an "n" value?
nf11=nf10['n'].count()
print("Roles with n Value : ",nf11)

# Q 14. Display the cast of "North by Northwest" in their correct "n"-value order, ignoring roles that did not earn a numeric "n" value.
nf12=cast.loc[['North by Northwest']]
nf12.isna().sum()
nf12.dropna(inplace=True) 
nf13=nf12.sort_values(by='n')
nf13.filter(items=['name','character','n'])


# Q15.Display the entire cast, in "n"-order, of the 1972 film "Sleuth"
nf14=cast.loc[(cast['title']=='Sleuth') & (cast['year']==1972)]
nf14.head()
nf15=nf14.sort_values(by='n')
nf15.filter(items=['title','year','name','character','n'])


# Q16. Now display the entire cast, in "n"-order, of the 2007 version of "Sleuth".
nf16=cast.loc[(cast['title']=='Sleuth') & (cast['year']==2007)]
nf16.head()
nf16.isna().sum()
nf16.dropna(inplace=True) 
nf17=nf16.sort_values(by='n')
nf17.filter(items=['title','year','name','character','n'])


# Q17. How many roles were credited in the silent 1921 version of Hamlet?
nf18 = cast.loc[(cast['title']=='Hamlet') & (cast['year']==1921)]
nf18['character'].count()

# Q18. How many roles were credited in Branagh’s 1996 Hamlet?
nf19 = cast.loc[(cast['title']=='Hamlet') & (cast['year']==1996)]
nf19['character'].count()


# Q19 How many "Hamlet" roles have been listed in all film credits through history?
nf20 = cast.loc[cast['character']=='Hamlet']
nf20['character'].count()


# Q20 How many people have played an "Ophelia"?
nf21 = cast.loc[cast['character']=='Ophelia']
nf21['character'].count()

# Q21 How many people have played a role called "The Dude"?
nf22 = cast.loc[cast['character']=='The Dude']
nf22['character'].count()


# Q22 How many people have played a role called "The Stranger"?
nf23 = cast.loc[cast['character']=='The Stranger']
nf23['character'].count()


# Q23 How many roles has Sidney Poitier played throughout his career?
nf24 = cast.loc[cast['name']=='Sidney Poitier']
nf24['name'].count()



# Q24 . How many roles has Judi Dench played?
nf25 = cast.loc[cast['name']=='Judi Dench']
nf25['name'].count()


# Q25 . List the supporting roles (having n=2) played by Cary Grant in the 1940s, in order by year.
nf26 = cast.loc[(cast['name']=='Cary Grant') &(cast['year'] >= 1940) & (cast['year'] <1950)&(cast['n']== 2)]
nf30=nf26.sort_values('year')
nf30.filter(items=['title','year','name','character','n'])


# Q 26 List the leading roles that Cary Grant played in the 1940s in order by year.
nf27 = cast.loc[(cast['name']=='Cary Grant') &(cast['year'] >= 1940) & (cast['year'] <1950)&(cast['n']== 1)]
nf31=nf27.sort_values('year')
print(nf31)
#nf31.filter(items=['title','year','name','character','n'])


# Q 27. How many roles were available for actors in the 1950s?
nf28 = cast.loc[(cast['type']=='actor') &(cast['year'] >= 1950) & (cast['year'] <1960)]
nf28['type'].count()

# Q 28. How many roles were available for actresses in the 1950s?
nf29 = cast.loc[(cast['type']=='actress') &(cast['year'] >= 1950) & (cast['year'] <1960)]
nf29['type'].count()


# Q29. How many leading roles (n=1) were available from the beginning of film history through 1980?
nf32 = cast.loc[(cast['n']== 1) & (cast['year'] <1980)]
nf32['n'].count()


# Q30. How many non-leading roles were available through from the beginning of film history through 1980?
nf33 = cast.loc[(cast['n'] != 1) & (cast['year'] <1980)]
nf33['n'].count()


# Q31 . How many roles through 1980 were minor enough that they did not warrant a numeric "n" rank?
nf34 = cast.loc[(cast['year'] >= 1980) & (cast['year'] <1990)]
nf34['n'].isna().count()

















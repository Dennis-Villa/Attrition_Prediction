# Analysis and Results Report

I show here the most important information I obtained after analyzing the relationship of each column of the dataset with attrition.

The complete analysis can be seen here:

- [The columns](./Data_Analysis.ipynb)
- [The model](../Model_Training/Model_Pipeline_Training.ipynb)

## Multi Variant Analysis

Since attrition in this case is presented as a binary value of Yes or No, I use a multivariate analysis with each variable.

### Columns Without Analytical Value

`EmployeeCount`, `Over18` and `StandardHours` have a single value for all employees in the data, so they are not valid columns for analysis.

### Important Columns

#### Age

![Attrition by Age](<./Images/Age.png>)

The first graph shows how the majority of employees are in the 25 to 50 age group. The second shows that the majority of attritions occur among younger employees, under 20 years of age, and that this decreases with increasing age, with a peak at around 60 years of age. This behavior is probably due to the fact that young people tend to explore various job options before settling in and older people tend to retire.

#### Environment Satisfaction

![Attrition by Environment Satisfaction](<./Images/Env_Satisfaction.png>)

The first graph shows how most employees are satisfied with their work environment. The second graph does not show much variation in attrition, except for a small peak in the lowest satisfaction value, as employees tend to stay in environments that are satisfying to them.

#### Job Satisfaction

![Attrition by Job Satisfaction](<./Images/Job_Satisfaction.png>)

The first graph shows how most employees are satisfied with their work, but the difference is not very large. Meanwhile, the second graph shows a slight increase in attrition at the minimum level and a slight decrease in attrition at the maximum level, indicating that workers remain in jobs in which they are satisfied.

#### Percent Salary Hike

![Attrition by Percent Salary Hike](<./Images/Pcnt_Salary_Hike.png>)

The first graph shows that the most common increase for employees was between 11% and 14%. The second graph shows a relatively constant value of attrition, with a peak at the 25% increase value, which could indicate that employees with a large salary increase tend to seek higher positions in the labor market.

#### Total Working Years

![Attrition by Total Working Years](<./Images/Tot_Work_Years.png>)

The first graph shows how the most common value of employees' years worked is between 5 and 10 years. The second shows that most attrition occurs among employees with less than 3 years of work, and that this decreases as the number of years increases, with a small peak at years 33 and 34 and a large peak at year 40. This behavior is probably due to the fact that people starting to work tend to explore several job options before settling down and older people tend to retire.

#### Years At Company

![Attrition by Years At Company](<./Images/Years_At_Company.png>)

The first graph shows how the most common value of years in the company for employees is between 1 and 10 years. The second shows that most departures occur among employees with less than 3 years of employment, and that this decreases as the number of years increases, with peaks at years 23, 31 and 32 and a large peak at year 40. This behavior is probably due to the fact that people who start working have a process where they adapt and decide to stay and older people tend to retire. I do not know the reason for the peak at year 23, so further research is needed.

#### Years With Current Manager

![Attrition by Years With Current Manager](<./Images/Years_Curr_Manager.png>)

The first graph shows how the most common values of years with the employees' current manager are 0, 2.5 and 7.5 years. The second shows that most departures occur among employees with a new manager, and that this decreases as the number of years increases, with a peak at 14 years. This behavior is probably due to the fact that the starting point includes new workers, who have already been shown to have a high probability of leaving. In addition, a new manager may be a reason for the departure of some workers due to a change in management policies. I don't know what the spike in the 14 years is due to.

#### Work Life Balance

![Attrition by Work Life Balance](<./Images/Work_Life_Balance.png>)

The first graph shows that most employees have a good work-life balance. The second shows that employees with a low level of work-life balance are more likely to attrition. In addition, a small increase in the probability of leaving is observed for employees with a maximum level of work-life balance.

#### Monthly Income

![Attrition by Monthly Income](<./Images/Monthly_Income.png>)

The first graph shows how most employees earn less than Rs. 9000 per month. The second shows that most attrition occurs among employees with a lower salary, and that this decreases as the salary increases, with a peak rise at Rs. 15000. This behavior is probably due to the fact that a low salary is an important incentive to change jobs. The peak at Rs. 15,000 is likely due to a change in the worker's quality of life, which leads him or her to seek new, better-paying options.

#### Business Travel

![Attrition by Business Travel](<./Images/Business_Travel.png>)

The first graph shows that most employees travel rarely. The second shows that employees who do not travel have the lowest attrition value and those who travel frequently have the highest value. This seems to indicate that employees prefer to avoid business travel.

#### Marital Status

![Attrition by Marital Status](<./Images/Marital_Status.png>)

The first chart shows that the most common marital status among employees is married. The second chart shows that single employees have the highest attrition rates, while the other two have similar percentages. This may be because single employees have more freedom with fewer family responsibilities.

#### Department

![Attrition by Department](<./Images/Department.png>)

The first graph shows that the majority of employees work in Research and Development or Sales. The second graph shows that Human Resources employees have the highest attrition rates, while the other two have similar percentages to each other. This seems to indicate that there is a worse work environment in the Human Resources department than in the others.

#### Education Field

![Attrition by Education Field](<./Images/Education_Field.png>)

The first chart shows that the majority of employees studied Life Sciences or Medical Sciences. The second chart shows that employees who studied Human Resources have the highest attrition rates, which corresponds with the previous result. The other fields have similar percentages to each other.

#### Average Daily Hours Worked

![Attrition by Average Daily Hours Worked](<./Images/Avg_Daily_H_Work.png>)

The first chart shows that most employees work an average of 6 to 8 hours daily. The second chart shows that the attrition rate increases as the number of working hours increases, peaking between 10 and 11 hours. This is likely due to employees with a higher workload leaving the company in search of better working conditions.

#### Average Weekly Hours Worked

![Attrition by Average Weekly Hours Worked](<./Images/Avg_Weekly_H_Work.png>)

Both charts behave similarly to the previous ones. The first chart shows that most employees work an average of 28 to 38 hours weekly. The second chart shows that the attrition rate increases as the number of working hours increases.

#### Days of Absence from Work

![Attrition by Days of Absence from Work](<./Images/Work_Days_Out.png>)

The first graph shows that most employees took between 18 and 32 days off during the year. The second graph shows that the rate of sick leave tends to decrease as the number of days off increases, without showing significant peaks, which seems to indicate that employees prefer work environments where they can take frequent breaks and days off.

## Correlation Analysis

By studying the correlation of the main columns, the impact they have on dropout can be determined, in order to know more clearly which are the most significant causes.

![Correlation Matrix](<./Images/Correlation_Matrix.png>)

This analysis indicated that the columns most related to attrition are average daily and weekly work hours, marital status, years with current manager, total years of work, age, and years in the company.

## Analysis results

The analysis showed that of the variables that most influence employee attrition, the main causes are the high number of working hours.
Precisely this statistic can be easily solved by the organization by establishing better work policies for its employees.

## Initial Outcome of the Attrition Model

![Model Results](<./Images/Model_Results.png>)

Analysis of the model created to predict employee attrition shows that it is able to predict with 85% accuracy the employees who will leave, and with almost 100% accuracy the employees who will stay with the company.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



    


def main():
    #load dataset into a dataframe
    credit_cards = pd.read_csv("D:\dm-group-1\Dataset\credit_default_train.csv")

   # credit_cards = pd.read_csv("/home/daniele/dm-group-1/Dataset/credit_default_train.csv")
    #remember: load the corresponding function from Riccardo's scripts
    credit_cards = remove_missing_values(credit_cards)
    
    column_names = credit_cards.columns
    #ba = balance[Continuous]
    columns_ba = ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep"]
    #pa = expense[Continuous]
    columns_pa = ["pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"]
    #ps = -x on time, x months in advance after which the credit was reimbursed [DISCRETE VALUE!] 
    columns_ps = ["ps-apr", "ps-may", "ps-jun", "ps-jul", "ps-aug", "ps-sep"]
    #firstly, create a data frame where we have three extra columns: ba, pa, ps
    #such attributes are the average values of the corresponding 6 attributes in the original data frame
    credit_cards_avg = create_data_frame_avg(credit_cards, columns_ba, columns_pa, columns_ps)
    
    pd.set_option('display.max_columns', 30)
    
    #filename  = open("outputfile",'w')
    #sys.stdout = filename
    
    #get some basic statistics about the attributes
    print(credit_cards_avg.describe())
    
    #remember that correlation doesn't make sense on class attributes!
    
    ######CORRELATION!!
    
    compute_correlation_between_attributes(credit_cards_avg, ["ba", "pa", "limit", "age"])
    
    compute_correlation_between_attributes(credit_cards_avg, columns_ba)
        
    compute_correlation_between_attributes(credit_cards_avg, columns_pa)
    
    compute_correlation_between_attributes(credit_cards_avg, columns_ps)
    
    #the only significant correlation (0.37) appears to be between limit and pa, and limit and ba (0.31)
    
    size = 12
    #clearly, the BA columns are strongly correlated to one another, but the PA columns are not very much correlated to PAs
    compute_correlation_between_attributes(credit_cards_avg, ["ba-apr", "ba-may", "ba-jun", "ba-jul", "ba-aug", "ba-sep", "pa-apr", "pa-may", "pa-jun", "pa-jul", "pa-aug", "pa-sep"])
        
    
    plot_average_all_months(12, credit_cards, columns_pa, "Average Payment Amount over Months", "pa")
    
    plot_average_all_months(12, credit_cards, columns_ba, "Average Bill Amount over Months", "ba")
    
    plot_average_ba_pa_all_months(size, credit_cards, columns_ba, columns_pa, columns_ps)     
    
    plot_attribute_group_ba_pa(credit_cards_avg, columns_ba, columns_pa, ("blue", "purple", "red", "orange"), size, "education")    
    
    plot_attribute_group_ba_pa(credit_cards_avg, columns_ba, columns_pa, ("blue", "red"), size, "credit_default")    

    plot_attribute_group_ba_pa(credit_cards_avg, columns_ba, columns_pa, ("blue"), size, "age") 
    
    plot_attribute_group_ba_pa(credit_cards_avg, columns_ba, columns_pa, ("blue", "purple", "red" ), size, "status")    
       
    #now try to make a scatter plot between the expenses and the balance
    #scatter_plot_attribute_columns(credit_cards_avg["pa"], credit_cards_avg["ba"], True)
    attributes = ['age', 'pa', 'ba', 'limit']
    plot_scatter_matrix(credit_cards_avg, size, attributes) 
    
    #plot the relation between two attributes and separate the ones that defaulted from the ones that didn't default
    plot_credit_default_attribute(credit_cards_avg, "age", "pa", size)
    
    plot_credit_default_attribute(credit_cards_avg, "age", "ba", size)
    
    plot_credit_default_attribute(credit_cards_avg, "age", "ps", size)
        
    plot_credit_default_attribute(credit_cards_avg, "age", "limit", size)
    
    #plot how many unique occurrences are of a certain type
    plot_count_per_discrete_attribute(credit_cards_avg, size)
    
    plot_ps_count(credit_cards_avg, size)
    
    #take our continuous attributes and use a histogram to represent their distribution
    plot_histogram_per_attribute(credit_cards_avg, size)
    
    
    #just an example, can also run it with "ps-jun", "ps-jul", "ps-aug", ...
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "ps-apr")
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "ps-may")
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "ps-jun")
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "ps-jul")
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "ps-aug")
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "ps-sep")


    
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "education")
    
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "sex")
    
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "age")

    plot_crosstab_credit_default_ps(credit_cards_avg, size, "status")
    
    plot_crosstab_credit_default_ps(credit_cards_avg, size, "limit")

    
    plot_group_histogram_attribute(credit_cards_avg, 'age', size, 1)
    
    plot_group_histogram_attribute(credit_cards_avg, 'pa', size, 500)
    
    
    plot_distribution(credit_cards_avg, "education", "age", 5)
    
    
    plot_distribution(credit_cards_avg, "status", "age", 5)

    plot_distribution(credit_cards_avg, "credit_default", "age", 5)
    
    plot_distribution(credit_cards_avg, "credit_default", "pa", 5)
    
    plot_distribution(credit_cards_avg, "credit_default", "ba", 5)
    
    plot_distribution(credit_cards_avg, "credit_default", "ps", 5)
    
    plot_distribution(credit_cards_avg, "credit_default", "limit", 5)
    
    credit_cards_avg = create_pa_ba_difference_column(credit_cards_avg)
    
    plot_distribution(credit_cards_avg, "credit_default", "ba-pa", 5)
    

    #Show the correlation matrix down here
    credit_cards_education = convert_education_to_numerical_attribute(credit_cards_avg)
    
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.width', 2000)
    
    credit_cards_education.drop(['ba', 'pa', 'ps'], axis=1, inplace=True)


    credit_cards_education.corr()

    
def plot_group_histogram_attribute(credit_cards_avg, attribute, size, bin_size):
        
    df = credit_cards_avg
    
    # Set up a grid of plots
    fig, axes = plt.subplots(2, 1, figsize=(10,10))
    
    # Histogram of AgeFill segmented by Survived
    df1 = df[df['credit_default'] == 'yes'][attribute]
    df2 = df[df['credit_default'] == 'no'][attribute]
    max_age = max(df[attribute])
    axes[0].hist([df1, df2], 
                 bins=int(max_age / bin_size), # bin_size
                 range=(1, max_age), 
                 stacked=True)
    axes[0].legend(('Default ', 'No Default'), loc='best')
    axes[0].set_title('Defaults by Age Groups Histogram')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    
    # Scatter plot Survived and AgeFill
    axes[1].scatter(df['credit_default'], df[attribute])
    axes[1].set_title('Credit Default by Age Plot')
    axes[1].set_xlabel('Defaults')
    axes[1].set_ylabel('Age')
    
    fig.savefig("/home/daniele/dm-local/credit_default_" + attribute + ".pdf")
    
    
def plot_crosstab_credit_default_ps(credit_cards_avg, size, attribute):
    
    df = credit_cards_avg

    crosstab = pd.crosstab(df[attribute], df['credit_default'])
    # Normalize the cross tab to sum to 1:
    crosstab_normalized = crosstab.div(crosstab.sum(1).astype(float), axis=0)
    
    crosstab_normalized.plot(kind='bar', stacked=True, 
                   title='Default by ' + str(attribute) + ' class')
    
    plt.savefig("/home/daniele/dm-local/ps-default" + str(attribute) + ".pdf")

    
    


def plot_ps_count(credit_cards_avg, size):
     # Set up a grid of plots
    fig = plt.figure(figsize=(size, size)) 
    fig_dims = (2, 3)
    
    df = credit_cards_avg
    
    plt.subplot2grid(fig_dims, (0, 0))
    df['ps-apr'].value_counts().plot(kind='bar', 
                                       title="Counts of ps-apr")
    
    plt.subplot2grid(fig_dims, (0, 1))
    df['ps-may'].value_counts().plot(kind='bar', 
                                       title="Counts of ps-may")
    
    plt.subplot2grid(fig_dims, (0, 2))
    df['ps-jun'].value_counts().plot(kind='bar', 
                                       title="Counts of ps-jun")
    
    plt.subplot2grid(fig_dims, (1, 0))
    df['ps-jul'].value_counts().plot(kind='bar', 
                                       title="Counts of ps-jul")
    
    plt.subplot2grid(fig_dims, (1, 1))
    df['ps-aug'].value_counts().plot(kind='bar', 
                                       title="Counts of ps-aug")
    
    plt.subplot2grid(fig_dims, (1, 2))
    df['ps-sep'].value_counts().plot(kind='bar', 
                                       title="Counts of ps-sep")
    
    
    
    plt.savefig("/home/daniele/dm-local/count_ps_plot.pdf")
    
    plt.show()
    


def plot_histogram_per_attribute(credit_cards_avg, size):
    fig = plt.figure(figsize=(size, size)) 
    fig_dims = (2, 2)
    
    df = credit_cards_avg

    plt.subplot2grid(fig_dims, (0, 0))
    
    plt.hist(df['age'], bins=20, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')
    plt.title('Age Histogram')
    
    
    
    plt.subplot2grid(fig_dims, (0, 1))
    plt.hist(df['ba'], bins=15, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')
    plt.title('Average Billing Amount Histogram')
    
    
    plt.subplot2grid(fig_dims, (1, 0))
    plt.hist(df['pa'], bins=150, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')
    plt.xlim(0, 40000)
    plt.title('Average Payment Amount Histogram')
    
    
    
    plt.subplot2grid(fig_dims, (1, 1))
    
    plt.hist(df['limit'], alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')
    plt.title('Average Limit Histogram')
    
    plt.title('Limit Histogram')
    
    plt.savefig("/home/daniele/dm-local/age_ba_pa_limit_hist.pdf")

    
    
    
    





def plot_count_per_discrete_attribute(credit_cards_avg, size):
    # Set up a grid of plots
    fig = plt.figure(figsize=(size, size)) 
    fig_dims = (2, 2)
    
    df = credit_cards_avg
    
    plt.subplot2grid(fig_dims, (0, 0))
    df['credit_default'].value_counts().plot(kind='bar', 
                                       title='Counts of Credit Defaults')
    plt.xticks(rotation=0)

    
    plt.subplot2grid(fig_dims, (0, 1))
    df['status'].value_counts().plot(kind='bar', title="Counts of Status")
    plt.xticks(rotation=0)

    
    plt.subplot2grid(fig_dims, (1, 0))
    df['sex'].value_counts().plot(kind='bar', title='Gender Counts')
    plt.xticks(rotation=0)
    
    plt.subplot2grid(fig_dims, (1, 1))
    df['education'].value_counts().plot(kind='bar', title='Education Counts')
    plt.xticks(rotation=0)

    
    
    plt.savefig("/home/daniele/dm-local/counts_plot.pdf")
    
    plt.show()
    

    

    
        
    
    
    
def plot_credit_default_attribute(credit_cards_avg, attribute_x, attribute_y, size):
     #and now let's plot some more relevant stuff, like who didn't repay based on different factors
    df = credit_cards_avg
      #let's show in green the ones that didn't default
    plt.scatter(df[df['credit_default'] == 'no'][attribute_x], 
            df[df['credit_default'] == 'no'][attribute_y], color='b', marker='*')
    
    #let's show in red the ones that defaulted
    plt.scatter(df[df['credit_default'] == 'yes'][attribute_x], 
            df[df['credit_default'] == 'yes'][attribute_y], color='r')
    
    plt.title("Scatter plot of the " + attribute_x + " and " + attribute_y + " attributes")
    
    
    plt.legend(('credit_default == yes', 'credit_default == no'))    

    plt.savefig("/home/daniele/dm-local/scatter_plot_" + attribute_x + "_" + attribute_y + ".pdf")

    
    
    
    
def scatter_plot_attribute_columns(column_x, column_y, log_notation):
    #compute the mean value of each columns' group
    
    if(log_notation == True):
        column_x = np.log(column_x)
        column_y = np.log(column_y)
        
    fig = plt.scatter(x = column_x, y = column_y)
    




def plot_scatter_matrix(credit_cards_avg, size, attributes):
     #let's try to plot a scatter matrix now for different attributes!

    pd.scatter_matrix(frame = credit_cards_avg[attributes], figsize  = [size, size])
    

    plt.savefig("/home/daniele/dm-local/scatter_matrix.pdf")

    
    
    
    
    
def create_data_frame_avg(credit_cards, columns_balance, columns_pa, columns_ps):
    mean_ba_column = credit_cards[columns_balance].mean(axis=1)
    mean_pa_column = credit_cards[columns_pa].mean(axis=1)
    mean_ps_column =  credit_cards[columns_ps].mean(axis=1)
    
    credit_cards["ps"] = mean_ps_column
    credit_cards["pa"] = mean_pa_column
    credit_cards["ba"] = mean_ba_column
    
    return(credit_cards)
    
    
def compute_correlation_between_attributes(credit_cards_avg, input_attributes):
    credit_cards_columns = credit_cards_avg[input_attributes]
    
    return credit_cards_columns.corr()
        



     
    

def plot_attribute_group_ba_pa(credit_cards, columns_ba, columns_pa, colors, size, attribute):
    figurePrint = plt.figure(figsize=(size, size)) 
    fig_dims = (2, 2)
    
    #Compute balance in the bank account based on the education level
    #Expected dependency: the higher the degree, the higher the balance 
    #Actual dependency: not completely true! People that have gone to graduate school may have lots of loans to pay back
    #so their balance is actually smaller than people who attended university
    compute_average_value_given_attribute_and_columns(credit_cards, columns_ba, attribute, colors, fig_dims, (0,0), ("Average Bill Amount wrt. " + attribute))
    
    #The fact above is further confirmed by the fact that the average age of people in the dataset is 32 y.o.
    #so, they haven't had much time to "accumulate" money

    
    #Compute salary of people based on their education level.
    #Expended dependency: the higher the degree, the higher the salary
    #Actual dependency: this is true. In fact, people attending graduate school specialize in a field and 
    #consequently earn a higher salary, on average. 
    compute_average_value_given_attribute_and_columns(credit_cards, columns_pa, attribute, colors, fig_dims, (0,1), ("Average Payment Amount wrt. " + attribute))
    
    #compute_average_value_given_attribute_and_columns(credit_cards, columns_ps, attribute, colors, fig_dims, (1,0), "Customer Repayment wrt. Education")
    
    figurePrint.savefig("/home/daniele/dm-local/" + attribute + "_plot.pdf", bbox_inches='tight')

    
    

#Input: data_frame: A Data frame loaded with data loaded from a dataset
#       columns_df: A set of columns taken from the data frame, containing a set of months 

#example call: 
#compute_average_value_given_attribute_and_columns(credit_cards, columns_balance, "education")
def compute_average_value_given_attribute_and_columns(credit_cards, columns_df, attribute, colors, fig_dims, coordinates, title):
    #firstly, group by education type
    group_attribute = credit_cards.groupby(attribute)
    
    #get the mean value of all rows in each column[just columns, no rows]
    mean_attribute_value = group_attribute.mean()
    
    #now perform another group by according to the unique values of the attribute column passed
    #ex: group by the different education types if passing "education"
    grouped_values_by_attribute = mean_attribute_value[columns_df]
    
   #let's now sum all the columns in the dataframe
   
    grouped_values_by_attribute['mean'] = grouped_values_by_attribute[columns_df].sum(axis=1) / len(columns_df)
   
    values_mean_y_axis = list(grouped_values_by_attribute['mean'])
    
    values_mean_x_axis = credit_cards[attribute].unique()
   
    plt.subplot2grid(fig_dims, coordinates)
    plt.title(title)
     
    return(plt.bar(x=values_mean_x_axis, height=values_mean_y_axis, color=colors))
     
    
    
def plot_average_all_months(size, credit_cards, columns_input, title, file_name):
    figurePrint = plt.figure(figsize=(size, size)) 
    plot_avg_single_plot(columns_input, credit_cards, title, plt)
    plt.savefig("/home/daniele/dm-local/average_" + file_name + "_months.pdf", bbox_inches='tight')
        
        
    
def plot_average_ba_pa_all_months(size, credit_cards, columns_ba, columns_pa, columns_ps):
    figurePrint = plt.figure(figsize=(size, size)) 
    fig_dims = (2, 2)

    #Show the April - September average balance of a person'
    plot_avg_ba_months = plot_average_given_columns(columns_ba, credit_cards, fig_dims, (0,0),"Customer Balance over months", plt)
    
     #Show the April - September average revenue of a person (fatturato?)
    plot_avg_pa_months = plot_average_given_columns(columns_pa, credit_cards, fig_dims, (0,1), "Customer Payment over months", plt)
    
    #Show the April - September average PS of a person 
    #plot_avg_ps_months = plot_average_given_columns(columns_ps, credit_cards, fig_dims, (1,0), "Customer Debt Repayment over months", plt)
    
    figurePrint.savefig("/home/daniele/dm-local/average_months.pdf", bbox_inches='tight')
        
        
        
    
def plot_avg_single_plot(list_columns, credit_cards, title, plt):    
    mean_list = []
    for column in list_columns:
        mean_column = credit_cards[column].mean()
        mean_list.append(mean_column)
    months = ["April", "May", "June", "July", "August", "September"]
    plt.title(title)
    return (plt.scatter(x=months, y = mean_list))
        

def plot_average_given_columns(list_columns, credit_cards, fig_dims, coordinates, title, plt):    
    mean_list = []
    for column in list_columns:
        mean_column = credit_cards[column].mean()
        mean_list.append(mean_column)
    months = ["April", "May", "June", "July", "August", "September"]
    plt.subplot2grid(fig_dims, coordinates)
    plt.title(title)
    return (plt.scatter(x=months, y = mean_list))
        
 
#Mean and standard deviation
def compute_mean_std_for_columns(list_columns, data_frame):
    mean_list = []
    std_list = []
    for column in list_columns:
        mean_column = data_frame[column].mean()
        std_column = data_frame[column].std()
        mean_list.append(mean_column)
        std_list.append(std_column)
    
    overall_mean = sum(mean_list) / len(mean_list)
    overall_std = sum(std_list) / len(std_list)
    
    print("Overall mean value of input columns is " + str(overall_mean))
    print("Mean STD value of input columns is " + str(overall_std))

    
    
    
#Riccardo's function

def remove_missing_values(df_in):
    df_out = df_in
    #SEX
    #fill missing values with the mode of sex (female)
    df_out['sex'] = df_out['sex'].fillna(df_out['sex'].mode()[0])
    
    #EDUCATION
    df_out['education'] = df_out['education'].fillna('others')
    
    #STATUS
    df_out['status'] = df_out['status'].groupby([df_out['sex'], df_out['education']]).apply( 
      lambda x: x.fillna(x.mode()[0]))
    
    #AGE
    #dataframe without rows where age is -1
    df_out['age'] = df_out['age'].groupby([df_out['sex'], df_out['education'],
      df_out['status']]).apply(lambda x: x.replace(-1, x.median()))
    
    return df_out



def plot_distribution(credit_cards, categorical_attribute, numerical_attribute, size):
    figurePrint = plt.figure(figsize=(size, size)) 

    df = credit_cards
    df['attrFill'] = df[numerical_attribute]
    
    attribute_types = sorted(df[categorical_attribute].unique())
    for education_type in attribute_types:
        df.attrFill[df[categorical_attribute] == education_type].plot(kind='kde')
        
    plt.xlabel(numerical_attribute)
    plt.legend(attribute_types, loc='best')
    plt.show()
    figurePrint.savefig("/home/daniele/dm-local/distribution_" + numerical_attribute + "_" + categorical_attribute + ".pdf", bbox_inches='tight')
    


#according to configuration 1:
#Ba(m), Pa(m+1)
def create_pa_ba_difference_column(credit_cards_avg):
    #we create a column for 
    #we wanna get how much the user had paid 
    
    #ex: we have a ba-apr of 30000 in ba-apr
    
    #ba-pay-may contains the rmeaining amount to pay for the month of april
    #remaining amount to pay = ba_apr - pa_may
    
    credit_cards_avg['ba-pa-apr'] = credit_cards_avg["pa-may"] - credit_cards_avg["ba-apr"] 
    credit_cards_avg['ba-pa-may'] = credit_cards_avg["pa-jun"] - credit_cards_avg["ba-may"]
    credit_cards_avg['ba-pa-jun'] = credit_cards_avg["pa-jul"] - credit_cards_avg["ba-jun"]
    credit_cards_avg['ba-pa-jul'] =  credit_cards_avg["pa-aug"] - credit_cards_avg["ba-jul"]
    credit_cards_avg['ba-pa-aug'] = credit_cards_avg["pa-sep"] - credit_cards_avg["ba-aug"]
    #now let's compute the average among these columns
    credit_cards_avg["ba-pa"] = credit_cards_avg['ba-pa-apr'] +  credit_cards_avg['ba-pa-may'] + credit_cards_avg["ba-pa-jun"] + credit_cards_avg["ba-pa-jul"] + credit_cards_avg["ba-pa-aug"]
    
    credit_cards_avg["ba-pa"] /= 5
    
    return credit_cards_avg

    
    
    
    
    
    
    







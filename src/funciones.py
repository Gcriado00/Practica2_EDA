def dame_variables_categoricas(dataset):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función clasificar_variables:
    ----------------------------------------------------------------------------------------------------------
        - Descripción : Funcion que recibe un dataset y devuelve una lista respectiva para cada tipo de variable
        (Categórica, Continua, Booleana y No clasificada)
        - Inputs:
            -- dataset : Pandas dataframe que contiene los datos
        - Return : 
            -- 1: la ejecución es incorrecta
            -- lista_var_bool: lista con los nombres de las variables booleanas del dataset de entrada, con valores
            unicos con una longitud de dos, que sean del tipo booleano y que presenten valores 'yes','no','n' & 'y' .
            -- lista_var_cat: lista con los nombres de las variables categóricas del dataset de entrada, con valores
            de tipo object o tipo categorical.
            -- lista_var_con: lista con los nombres de las variables continuas del dataset de entrada, con valores 
            de tipo float o con una longitud de valores unicos mayor a dos. 
            -- lista_var_no_clasificadas: lista con los nombres de las variables no clasificadas del dataset de 
            entrada, que no cumplen con los aspectos anteriormente mencionadas de las demás listas. 
    '''
    
    if dataset is None:
        # Resultante al no brindar ningun DataFrame
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    # Listas para cada tipo de variable
    lista_var_bool = []
    lista_var_cat = []
    lista_var_con = []
    lista_var_no_clasificadas = []
    
    for columna in dataset.columns:
        # Valores unicos por columna sin los NAs
        valores_unicos = dataset[columna].dropna().unique()
        # Trato de mayusculas
        valores_lower = set(val.lower() for val in valores_unicos if isinstance(val, str))
        
        # Variables booleanas
        if (len(valores_unicos) == 2 and
            (valores_lower <= {"yes", "no", "n", "y"} or
             set(valores_unicos) <= {0, 1} or 
             pd.api.types.is_bool_dtype(dataset[columna]))):
            lista_var_bool.append(columna)
        
        # Variables continuas
        elif pd.api.types.is_float_dtype(dataset[columna]) and len(valores_unicos) > 2:
            lista_var_con.append(columna)
        
        # Variables categóricas
        elif pd.api.types.is_object_dtype(dataset[columna]) or pd.api.types.is_categorical_dtype(dataset[columna]):
            lista_var_cat.append(columna)
        
        elif set(valores_unicos).issubset({1, 2, 3}):
            lista_var_cat.append(columna)
        
        # Variables no clasificadas
        else:
            lista_var_no_clasificadas.append(columna) 

    # Calcula la cantidad de cada tipo de variable
    c_v_b = len(lista_var_bool)
    c_v_ca = len(lista_var_cat)
    c_v_co = len(lista_var_con)
    c_v_f = len(lista_var_no_clasificadas)

    print("Variables Booleanas:", c_v_b, lista_var_bool)
    print('============================================================================================================================================================================')
    print("Variables Categóricas:", c_v_ca, lista_var_cat)
    print('============================================================================================================================================================================')
    print("Variables Continuas:", c_v_co, lista_var_con)
    print('============================================================================================================================================================================')
    print("Variables no clasificadas:", c_v_f, lista_var_no_clasificadas)

    return lista_var_bool, lista_var_cat, lista_var_con, lista_var_no_clasificadas

#-------------------------------------------------------------------------------------------

def plot_feature(df, col_name, isContinuous, target):
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    count_null = df[col_name].isnull().sum()
    if isContinuous:
        
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name+ ' Numero de nulos: '+str(count_null))
    plt.xticks(rotation = 90)


    if isContinuous:
        sns.boxplot(x=col_name, y=target, data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by '+target)
    else:
        data = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index() 
        data.columns = [i, target, 'proportion']
        #sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        ax2.set_ylabel(target+' fraction')
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()

#----------------------------------------------------------------------------------------------

def dame_variables_categoricas(dataset=None):
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    lista_variables_categoricas = []
    other = []

    for i in dataset.columns:

        if dataset[i].dtype == object:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)
        
        if dataset[i].dtype == int:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 10:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other

#-------------------------------------------------------------------------------

def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

#-------------------------------------------------------------------------------------

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('TARGET',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

#---------------------------------------------------------------------

def get_percent_null_values_target(pd_loan, list_var_continuous, target):

    pd_final = pd.DataFrame()
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('TARGET',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum()/pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

#--------------------------------------------------------------------

def cramers_v(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

#------------------------------------------------------------------------

def cramers_v(matrix):
    chi2, p, dof, ex = chi2_contingency(matrix)  # Chi-squared test
    return np.sqrt(chi2 / (matrix.sum().sum() * (min(matrix.shape) - 1)))


















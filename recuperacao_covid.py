import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from pandas_profiling.profile_report import ProfileReport

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def fixing_data_type(column, first_symbol, second_symbol):
    try:
        column = column.apply(lambda x : str(x).replace(first_symbol, second_symbol))
        column = pd.to_datetime(column, format='%Y%m%d')
    except ValueError:
        column = column.apply(lambda x : str(x).replace(first_symbol, second_symbol))
        column = pd.to_datetime(column, dayfirst=True)
    return column


def drop_columns(df, columns):
    df.drop(columns=columns, inplace=True)
    return df


def plot_chart(desired_variable, df, order, title=None, xlabel=None, ylabel=None,
               hue=None, ax=None):
    sns.countplot(y=desired_variable,
                  hue=hue,
                  data=df,
                  palette='rocket',
                  order=order,
                  ax=ax).set_title(title)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.grid(axis='x', linestyle='--')
    plt.minorticks_on()


def plot_box_plot(column, title=None, xlabel=None, ax=None):
    sns.boxplot(column, palette='rocket', ax=ax).set_title(title)
    plt.xlabel(xlabel)
    plt.grid(axis='x', linestyle='--')
    plt.minorticks_on()


def plot_time_series(column, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    ax.hist(column, bins=50, color='#301934')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def save_fig_show(figname, image_path):
    plt.savefig(image_path + figname + '.jpeg')
    return plt.show()


path = 'C:/Users/nanda/PycharmProjects/puc_tcc/data/'
covid_sistema_um = pd.read_csv(f'{path}XLSX_Sistemas_1.txt', sep='\t')
covid_sistema_dois = pd.read_csv(f'{path}XLSX_Sistemas_2.txt', sep='\t')
covid_sistema_tres = pd.read_csv(f'{path}XLSX_Sistemas_3.txt', sep='\t')
covid_sistema_quatro = pd.read_csv(f'{path}xlsx_sistemas.txt', sep='\t')

covid_sistema_um['DATA_NOTIFICACAO'] = fixing_data_type(
    covid_sistema_um['DATA_NOTIFICACAO'], '-', '')
covid_sistema_dois['DATA_NOTIFICACAO'] = fixing_data_type(
    covid_sistema_dois['DATA_NOTIFICACAO'], '-', '')
covid_sistema_tres['DATA_NOTIFICACAO'] = fixing_data_type(
    covid_sistema_tres['DATA_NOTIFICACAO'], '-', '')
covid_sistema_quatro['DATA_NOTIFICACAO'] = fixing_data_type(
    covid_sistema_quatro['DATA_NOTIFICACAO'], '/', '-')

covid_df = pd.concat([covid_sistema_um, covid_sistema_dois, covid_sistema_tres,
                      covid_sistema_quatro])

print(covid_df.info())

covid_df.reset_index(drop=True, inplace=True)
report = ProfileReport(covid_df, title='Primeira investigação do dataset')
report.to_file('TCC_COVID_REPORT_1.html')

# TRATAMENTO DOS DADOS
print(covid_df.isnull().sum())

drop_columns(covid_df, ['ID', 'CODIGO', 'DATA_EVOLUCAO', 'ETNIA', 'MUNICIPIO_RESIDENCIA',
                        'MICRO', 'DATA_1_SINTOMA', 'DATA_ATUALIZACAO',
                        'CLASSIFICACAO_CASO'])

print(f"Searching for the outlier:\n"
      f"{covid_df['DATA_NOTIFICACAO'].dt.year.value_counts()}")
print(covid_df.loc[covid_df['DATA_NOTIFICACAO'].dt.year == 1957])

covid_df = covid_df.loc[covid_df['DATA_NOTIFICACAO'].dt.year != 1957]
print(f"After deleting the outlier:\n"
      f"{covid_df['DATA_NOTIFICACAO'].dt.year.value_counts()}")

covid_df = covid_df.loc[covid_df['SEXO'] != 'NAO INFORMADO']

image_path = 'C:/Users/nanda/PycharmProjects/puc_tcc/images/'

plot_box_plot(covid_df['IDADE'], 'Age outlier', xlabel='Age')
save_fig_show('Age_outlier', image_path=image_path)

print(f"Looking for the outliers: "
      f"{covid_df.loc[covid_df['IDADE'] > 100, ['IDADE', 'FAIXA_ETARIA']]}")

covid_df = covid_df.loc[covid_df['IDADE'] <= 100]

print(covid_df.duplicated().sum())

covid_df.fillna(method='ffill', inplace=True)

print(covid_df.duplicated().sum())

plt.figure(figsize=(16, 8))
plot_chart(covid_df['MACRO'], covid_df, covid_df['MACRO'].value_counts().index,
           'Cases by Macroregion', xlabel='Cases', ylabel='Macroregion')
save_fig_show('Macroregion_cases', image_path)

plt.figure(figsize=(16, 8))
plot_chart(covid_df['URS'], covid_df, covid_df['URS'].value_counts().index,
           'Cases by URS', 'Cases', 'URS')
save_fig_show('Cases_by_urs', image_path)

print(f"Cases by Health Information Systems: \n"
      f"{covid_df['ORIGEM_DA_INFORMACAO'].value_counts()}")
plt.figure(figsize=(16, 8))
plot_chart('ORIGEM_DA_INFORMACAO', covid_df,
           covid_df['ORIGEM_DA_INFORMACAO'].value_counts().index,
           'Cases by Health Information Systems', 'Cases', 'Health Information Systems')
save_fig_show('Health_Information_Systems', image_path)

plt.figure(figsize=(16, 8))
plot_time_series(covid_df['DATA_NOTIFICACAO'], 'Date', 'Number of cases')
save_fig_show('Number_of_cases_over_time', image_path)

fig, ax = plt.subplots(1, 2, figsize=(27, 10))
plot_chart('FAIXA_ETARIA', covid_df,
           covid_df['FAIXA_ETARIA'].value_counts().sort_index().index,
           'Cases by Age Range', hue='SEXO', ax=ax[1])
plot_box_plot(covid_df['IDADE'], 'Age of patients', ax=ax[0])
ax[0].set_xlabel(xlabel='Age', fontsize=13)
ax[0].tick_params(axis='both', which='major', labelsize=13)
ax[0].grid(axis='x', linestyle='--')
ax[0].minorticks_on()
ax[1].tick_params(axis='both', which='major', labelsize=13)
ax[1].set_xlabel(xlabel='Cases', fontsize=13)
ax[1].set_ylabel(ylabel='Age Range', fontsize=13)
ax[1].grid(axis='x', linestyle='--')
ax[1].minorticks_on()
save_fig_show('Age', image_path)

plt.figure(figsize=(13, 5))
covid_df['COMORBIDADE'].value_counts()
plot_chart('COMORBIDADE', covid_df, covid_df['COMORBIDADE'].value_counts().index,
           'Presence of comorbidity', xlabel='Count', ylabel='Presence', hue='SEXO')
save_fig_show('Comorbidity', image_path)

plt.figure(figsize=(15, 8))
plot_chart('RACA', covid_df, covid_df['RACA'].value_counts().index, title='Cases by Race',
           ylabel='Race', xlabel='Cases', hue='SEXO')
save_fig_show('Race', image_path)

plt.figure(figsize=(13, 8))
plot_chart('INTERNACAO', covid_df, covid_df['INTERNACAO'].value_counts().index,
           title='Inpatients Cases', xlabel='Cases', ylabel='Inpatients', hue='SEXO')
save_fig_show('Inpatients', image_path)

plt.figure(figsize=(13, 8))
plot_chart('UTI', covid_df, covid_df['UTI'].value_counts().index, title='UTI Cases',
           xlabel='Cases', ylabel='UTI', hue='SEXO')
save_fig_show('UTI_cases', image_path)

plt.figure(figsize=(17, 8))
plot_chart('EVOLUCAO', covid_df, covid_df['EVOLUCAO'].value_counts().index,
           title='Cases by Evolution', xlabel='Case count', ylabel='Case evolution',
           hue='SEXO')
save_fig_show('Evolution_by_sex', image_path)

# Preparação dos dados para o modelo
drop_columns(covid_df, ['DATA_NOTIFICACAO', 'ORIGEM_DA_INFORMACAO', 'IDADE'])

encoded_df = pd.get_dummies(covid_df, columns=['SEXO', 'INTERNACAO', 'UTI', 'RACA',
                                               'MACRO', 'URS', 'COMORBIDADE'],
                            drop_first=True, dtype='int64')

faixa_idade_dict = {
    '<1ANO' : 1,
    '1 A 9 ANOS': 2,
    '10 A 19 ANOS': 3,
    '20 A 29 ANOS': 4,
    '30 A 39 ANOS': 5,
    '40 A 49 ANOS': 6,
    '50 A 59 ANOS': 7,
    '60 A 69 ANOS': 8,
    '70 A 79 ANOS': 9,
    '80 A 89 ANOS': 10,
    '90 OU MAIS': 11
}
evolucao_dict = {'RECUPERADO': 1, 'EM ACOMPANHAMENTO': 2, 'OBITO': 3}

encoded_df['FAIXA_IDADE_ORDINAL'] = encoded_df.FAIXA_ETARIA.map(faixa_idade_dict)
encoded_df['EVOLUCAO_ORDINAL'] = encoded_df.EVOLUCAO.map(evolucao_dict)
drop_columns(encoded_df, ['FAIXA_ETARIA', 'EVOLUCAO'])


corr = encoded_df.corr()
plt.figure(figsize=(40, 40))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
graph = sns.heatmap(corr, cmap=cmap, center=0, vmin=-1, vmax=1, annot=True, square=True,
                    linewidths=5, cbar_kws={'shrink': 0.5}, annot_kws={'size': 12})
graph.set_xticklabels(graph.get_xmajorticklabels(), fontsize=18)
graph.set_yticklabels(graph.get_ymajorticklabels(), fontsize=18)
plt.tight_layout()
plt.savefig('Correlation.jpeg')

y = encoded_df['EVOLUCAO_ORDINAL']
X = drop_columns(encoded_df, ['EVOLUCAO_ORDINAL'])

sampling_strategy_under = {1: 622033, 2: 116135, 3: 108466}
under = RandomUnderSampler(replacement=True, sampling_strategy=sampling_strategy_under,
                           random_state=42)
print('Original dataset shape %s' % Counter(y))
X_resampled, y_resampled = under.fit_resample(X, y)
print(f'Resampled dataset shape whith RandonUnderSample: {Counter(y_resampled)}.')

sampling_strategy_over = {1: 622033, 2: 622032, 3: 622032}
over = RandomOverSampler(sampling_strategy=sampling_strategy_over, random_state=42)
X_combined_sampling, y_combined_sampling = over.fit_resample(X_resampled, y_resampled)
print(f'Combined RandomOverSampling with RandonUnderSampling: '
      f'{Counter(y_combined_sampling)}')

# Modelando
# Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=25)
cv_results_tree = cross_validate(model, X_combined_sampling, y_combined_sampling,
                                 return_train_score=True)

# RandomForestClassifier
modelrf = RandomForestClassifier(criterion='entropy', random_state=42, max_depth=15)
cv_results_random = cross_validate(modelrf, X_combined_sampling, y_combined_sampling,
                                   return_train_score=True)

# Light Gradient Boost Machine
modelgbm = LGBMClassifier(max_depth=15, random_state=42)
cv_results_gbc = cross_validate(modelgbm, X_combined_sampling, y_combined_sampling,
                                return_train_score=True)

print(f'Decision Tree Classifier Cross-Validation Train: {cv_results_tree["train_score"]}\n'
      f'Decision Tree Classifier Cross-Validation Test: {cv_results_tree["test_score"]}\n'
      f'Random Forest Classifier Cross-Validation Train: {cv_results_random["train_score"]}\n'
      f'Random Forest Classifier Cross-Validation Test: {cv_results_random["test_score"]}\n'
      f'Gradient Boosting Classifier Cross-Validation Train: {cv_results_gbc["train_score"]}\n'
      f'Gradient Boosting Classifier Cross-Validation Test: {cv_results_gbc["test_score"]}')

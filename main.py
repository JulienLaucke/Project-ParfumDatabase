import base64
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image4.jpg")

#Erstellen des Hintergrundbildes
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-size: 100%;
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

[data-testid="element-container"]{{
style:"color-scheme: black";
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Daten einlesen
@st.cache_data
def load_data():
    df = pd.read_csv('df.csv')
    df2 = pd.read_csv('df2.csv')
    df3 = pd.read_csv('parfum_duftnoten_final.csv')
    new_df = pd.read_csv('new_df.csv')
    return df, df2, df3, new_df

df, df2, df3, new_df = load_data()

# Vorbereitung für das ML Modell
merged_df = df.merge(df3, on='ParfumIndex')
merged_df = merged_df.merge(df2, left_on='DuftnoteID', right_on='DuftnoteID')

dummy_columns = pd.get_dummies(merged_df['AllgemeineDuftnote'], prefix='AllgemeineDuftnote')
merged_df_with_dummies = pd.concat([merged_df, dummy_columns], axis=1)
parfum_features = merged_df_with_dummies.groupby('ParfumIndex')[dummy_columns.columns.tolist()].sum().reset_index()

df_sex = pd.merge(merged_df[['ParfumIndex', 'Geschlecht']], parfum_features, on='ParfumIndex', how='left')

X = df_sex.drop(['Geschlecht','ParfumIndex'], axis=1)
y = df_sex['Geschlecht']

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Funktion zur Ähnlichkeitsberechnung
def calculate_similarity(selected_indices, new_df):
    selected_notes = set(new_df[new_df['ParfumIndex'].isin(selected_indices)]['DuftnoteID'])
    similarities = new_df.groupby('ParfumIndex')['DuftnoteID'].apply(lambda x: len(selected_notes & set(x)) / len(selected_notes | set(x)))
    return similarities

# Funktion für die Parfumempfehlung 
def get_multiple_recommendations(selected_indices, new_df, gender_preference, num_recommendations=5):
    # Filter basierend auf Geschlecht
    if gender_preference == 'Männlich':
        filtered_df = new_df[new_df['Geschlecht'].isin(['Männer', 'Unisex'])]
    elif gender_preference == 'Weiblich':
        filtered_df = new_df[new_df['Geschlecht'].isin(['Frauen', 'Unisex'])]
    else:
        filtered_df = new_df

    similarities = calculate_similarity(selected_indices, filtered_df)
    similarities = similarities.drop(selected_indices)
    top_similar_indices = similarities.nlargest(num_recommendations).index
    top_similar_parfums = filtered_df[filtered_df['ParfumIndex'].isin(top_similar_indices)]

    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")  
    sns.barplot(x=top_similar_parfums['Parfum'].unique(), y=similarities[top_similar_indices].values, alpha=0.8, palette="viridis")
    plt.title('Parfümempfehlungen', color='white')
    plt.ylabel('Ähnlichkeit', color='white')
    plt.xlabel('Parfüm', color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('#333333')  
    plt.gcf().set_facecolor('#333333')  
    st.pyplot(plt)

    return top_similar_parfums

# Funktion für die Auswahl der Parfüms
def get_user_selected_indices(gender_preference):
    # Filter basierend auf Geschlecht
    if gender_preference == 'Männlich':
        available_parfums = df[df['Geschlecht'].isin(['Männer', 'Unisex'])]
    elif gender_preference == 'Weiblich':
        available_parfums = df[df['Geschlecht'].isin(['Frauen', 'Unisex'])]
    else:
        available_parfums = df

    selected_parfums = st.multiselect('Wählen Sie Parfümnamen aus:', available_parfums['Parfum'].unique())
    selected_indices = available_parfums[available_parfums['Parfum'].isin(selected_parfums)]['ParfumIndex'].tolist()
    return selected_indices

# Funktion für das ML Modell
def random_forest_model(df):
    X = df.drop(['Geschlecht', 'ParfumIndex'], axis=1)
    y = df['Geschlecht']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Genauigkeit des Modells: {accuracy:.2f}")

selected = option_menu(
    menu_title=None,
    options=["Home", "Visualisierung", "Empfehlung", "ML Modell"],
    icons=["house", "bar-chart", "chat-dots", "calculator-fill"],
    menu_icon="cast",
    orientation="horizontal"
)

if selected == "Home":
    st.title("ScentSense: Analyse und Empfehlung von Parfüms durch Data Science")
    st.write("""
        Diese Anwendung ermöglicht es Ihnen, verschiedene Aspekte von Parfümdaten zu erkunden.
        Sie können die Daten visualisieren, Empfehlungen basierend auf Duftnoten erhalten und
        maschinelle Lernmodelle verwenden, um Vorhersagen zu treffen. 
        Nutzen Sie das Menü oben, um zwischen den verschiedenen Funktionen zu navigieren.
    """)

if selected == "Visualisierung":
    st.title(f"{selected}")
    st.subheader("Wählen Sie die Visualisierung aus:")
    visualization_option = st.selectbox("", ["Durchschnittlicher Score pro Hersteller", "Häufigkeit der Parfümeure", "Beliebteste Duftnoten", "Score Normalverteilung", 
                                             "Duftnoten im Zeitverlauf", "Häufigster Hersteller", "Duftgruppen-Analyse", "Verteilung der Parfums nach Geschlecht"])

    if visualization_option == "Durchschnittlicher Score pro Hersteller":
        top_n = st.slider("Wählen Sie die Anzahl der Hersteller aus", min_value=1, max_value=50, value=20)
        durchschnittlicher_score_pro_hersteller = df.groupby('Hersteller')['Score'].mean().sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(10, 6))
        sns.set(style="darkgrid")
        sns.barplot(x=durchschnittlicher_score_pro_hersteller.index, y=durchschnittlicher_score_pro_hersteller.values, alpha=0.8, palette="viridis")
        plt.title('Durchschnittlicher Score pro Hersteller', color='white')
        plt.ylabel('Durchschnittlicher Score', color='white')
        plt.xlabel('Hersteller', color='white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)

    elif visualization_option == "Häufigkeit der Parfümeure":
        df_filtered = df[df['parfumeur'] != 'Nicht angegeben']
        haeufigste_parfumeure = df_filtered['parfumeur'].value_counts(ascending=False).head(30)
        plt.figure(figsize=(10, 6))
        sns.set(style="darkgrid")
        sns.barplot(x=haeufigste_parfumeure.index, y=haeufigste_parfumeure.values, alpha=0.8, palette="viridis")
        plt.title('Häufigkeit der Parfümeure', color='white')
        plt.ylabel('Anzahl der Parfums', color='white')
        plt.xlabel('Parfümeur', color='white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)

    elif visualization_option == "Beliebteste Duftnoten":
        top_n = st.slider("Wählen Sie die Anzahl der Duftnoten aus", min_value=1, max_value=50, value=10)
        
        beliebteste_duftnoten = new_df['Duftnote'].value_counts().head(top_n)
        plt.figure(figsize=(10, 6))
        sns.set(style="darkgrid")
        sns.barplot(x=beliebteste_duftnoten.index, y=beliebteste_duftnoten.values, alpha=0.8, palette="viridis")
        plt.title(f'Beliebteste Duftnoten (Top {top_n})', color='white')
        plt.ylabel('Anzahl der Parfums', color='white')
        plt.xlabel('Duftnote', color='white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)


        plt.figure(figsize=(8, 8))
        plt.pie(beliebteste_duftnoten.values, labels=beliebteste_duftnoten.index, startangle=140, textprops={'color': 'white'})
        plt.title(f'Beliebteste Duftnoten (Top {top_n})', color='white', pad=20)
        plt.axis('equal')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)


    elif visualization_option == "Score Normalverteilung":
        score = df['Score'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.set(style="darkgrid")
        sns.barplot(x=score.index, y=score.values, alpha=0.8, palette="viridis")
        plt.title('Parfums pro Erscheinungsjahr', color='white')
        plt.ylabel('Anzahl der Parfums', color='white')
        plt.xlabel('Erscheinungsjahr', color='white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)


    elif visualization_option == "Duftnoten im Zeitverlauf":
        min_year = int(df['Erscheinungsjahr'].min())
        max_year = int(df['Erscheinungsjahr'].max())

        year_range = st.slider(
            "Wählen Sie die Spanne von Jahren aus",
            min_year,
            max_year,
            (min_year, max_year))

        filtered_df = new_df[(new_df['Erscheinungsjahr'] >= year_range[0]) & (new_df['Erscheinungsjahr'] <= year_range[1])]

        df_duftstoffe = filtered_df['AllgemeineDuftnote'].apply(pd.Series)

        duftstoffe_nach_jahr = df_duftstoffe.melt(value_name='AllgemeineDuftnote').groupby([filtered_df['Erscheinungsjahr'], 'AllgemeineDuftnote']).size().reset_index(name='Häufigkeit')

        top_n = st.slider("Anzahl der angezeigten Duftstoffe", min_value=1, max_value=20, value=10)

        top_duftstoffe_nach_jahr = duftstoffe_nach_jahr.sort_values(['Erscheinungsjahr', 'Häufigkeit'], ascending=[True, False]).groupby('Erscheinungsjahr').head(top_n)

        relevant_duftstoffe = top_duftstoffe_nach_jahr[top_duftstoffe_nach_jahr['Häufigkeit'] > 3]
        num_duftstoffe = len(relevant_duftstoffe['AllgemeineDuftnote'].unique())

        fig = px.line(relevant_duftstoffe, x='Erscheinungsjahr', y='Häufigkeit', color='AllgemeineDuftnote')

        fig.update_layout(
            xaxis_title='Erscheinungsjahr',
            yaxis_title='Häufigkeit',
            plot_bgcolor='#333333',
            paper_bgcolor='#333333',
            font=dict(color='white'),
            width=1000,
            height=600,
            margin=dict(l=1), 
            legend=dict(
            traceorder="normal", 
            font=dict(color='white'),  
            bgcolor='#333333', 
            bordercolor='white', 
            borderwidth=1  
            )
        )

        for trace in fig.data:
            if trace.name not in relevant_duftstoffe['AllgemeineDuftnote'].unique():
                trace.visible = 'legendonly'

        st.plotly_chart(fig)

    elif visualization_option == "Häufigster Hersteller":
        top_n = st.slider("Wählen Sie die Anzahl der Hersteller aus", min_value=1, max_value=10, value=5)
        anzahl_parfums_pro_hersteller = df['Hersteller'].value_counts().head(top_n)

        fig = px.pie(anzahl_parfums_pro_hersteller, 
                    values=anzahl_parfums_pro_hersteller.values, 
                    names=anzahl_parfums_pro_hersteller.index, 
                    title=f'Häufigster Hersteller (Top {top_n})')

        fig.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig)

    elif visualization_option == "Duftgruppen-Analyse":
        duftgruppen_verteilung = new_df['Duftgruppen'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=duftgruppen_verteilung.index, y=duftgruppen_verteilung.values, palette="viridis")
        plt.title('Duftgruppen-Verteilung', color='white')
        plt.ylabel('Anzahl der Parfums', color='white')
        plt.xlabel('Duftgruppen', color='white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)
    
    elif visualization_option == "Verteilung der Parfums nach Geschlecht":
        start_year, end_year = st.slider("Zeitraum auswählen:", 1920, 2024, (1920, 2024))

        filtered_df = df[(df['Erscheinungsjahr'] >= start_year) & (df['Erscheinungsjahr'] <= end_year)]

        verteilung_geschlecht = filtered_df['Geschlecht'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=verteilung_geschlecht.index, y=verteilung_geschlecht.values, palette="viridis")
        plt.title('Verteilung der Parfums nach Geschlecht', color='white')
        plt.ylabel('Anzahl der Parfums', color='white')
        plt.xlabel('Geschlecht', color='white')
        plt.xticks(rotation=90, color='white')
        plt.yticks(color='white')
        plt.gca().set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        st.pyplot(plt)

if selected == "Empfehlung":
    st.title(f"{selected}")

    # Geschlechtspräferenz auswählen
    gender_preference = st.radio(
        'Welche Art von Parfüms suchen Sie?',
        ('Männlich', 'Weiblich', 'Unisex')
    )

    # Pass the gender preference to the function
    selected_indices = get_user_selected_indices(gender_preference)
    if selected_indices:
        top_similar_parfums = get_multiple_recommendations(selected_indices, merged_df, gender_preference)
        st.dataframe(top_similar_parfums[['Parfum', 'Hersteller']].drop_duplicates())
    else:
        st.write('Bitte wählen Sie mindestens ein Parfüm aus.')

if selected == "ML Modell":
    st.title("Vorhersage des Geschlechts basierend auf Duftnoten")

    selected_notes = st.multiselect('Wählen Sie Duftnoten aus:', df2['Duftnote'].unique())

    if st.button('Vorhersage berechnen'):
        if selected_notes:
            allgemeine_notes = df2[df2['Duftnote'].isin(selected_notes)]['AllgemeineDuftnote'].unique()
            note_columns = ['AllgemeineDuftnote_' + note for note in allgemeine_notes]
            
            input_data = pd.DataFrame(np.zeros((1, len(dummy_columns.columns))), columns=dummy_columns.columns)
            
            for note_col in note_columns:
                if note_col in input_data.columns:
                    input_data[note_col] = 1
            
            prediction = clf.predict(input_data)
            prediction_label = label_encoder.inverse_transform(prediction)

            if prediction == 0:
                gender = "weiblich"
            elif prediction == 1:
                gender = "männlich"
            else:
                gender = "unisex"

            st.write(f"Das Parfüm wird als {gender} klassifiziert.")
        else:
            st.write("Bitte wählen Sie mindestens eine Duftnote aus.")

   

import streamlit as st
import requests
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
import time
import datetime
import matplotlib.pyplot as plt
# import seaborn as sns
from digi_plotfuncs import *
from data_call import *

# Set page in wide mode
st.set_page_config(layout="wide")
# Set plotting style
plt.style.use("fivethirtyeight")

st.title("Digi Analysis")

last_call = pd.read_csv("app_data/last_call_date.csv", parse_dates=["last_call"])
st.sidebar.subheader("LAST DATA CALL:")
st.sidebar.write(f'Data up to date as of: {last_call.last_call.iloc[-1].strftime("%d-%m-%Y")}')
minutes = round((datetime.datetime.now() - last_call.last_call.iloc[-1]).seconds/60)
days = (datetime.datetime.now() - last_call.last_call.iloc[-1]).days
st.sidebar.write(f"{days} days and {minutes} minutes ago!")

# data = st.sidebar.button("Update Data")
# if data:
df_account, df_profile, df_family, df_task_definition, df_task_execution = call_data()
print(df_account.shape)
# Else call in from .csv





# view = st.sidebar.radio("Select a tool to view / call in data", ["Back End", "Front End", "Data Call", "Synthetic"])
view_ = st.sidebar.selectbox("Select dashboard", ["Downloads", "Accounts", "Membership", "Tasks", "Front End"])


if view_ == "Accounts":

    # Get family set for merge with task df later down sheet
    # account_db, account_server = call_db(database="digi-account")
    # family_df = pd.DataFrame(list(account_db["user-family"].find()))



    # account_df = pd.DataFrame(list(account_db["user-account"].find()))
    account_df = pd.read_csv("account_df.csv")
    account_df["creationTime"] = pd.to_datetime(account_df.creationTime)
    account_df["creationTime"] = account_df.creationTime.dt.strftime('%Y-%m-%d %H:%M:%S')
    account_df["creationTime"] = pd.to_datetime(account_df.creationTime)
    account_df["dayOfYear"] = account_df.creationTime.dt.dayofyear
    account_df["daysLive"] = account_df.dayOfYear - 65

    # print(account_df._id.nunique(), account_df.shape)
    account_df = account_df.sort_values("creationTime").reset_index(drop=True)
    account_df["creationDate"] = pd.to_datetime(account_df.creationTime.dt.date)

    full_period = pd.date_range(account_df.creationDate.min(), account_df.creationDate.max())
    full_period_df = pd.DataFrame(full_period, columns=["creationDate"])

    new_accounts_grouped = (account_df
                            .groupby(["creationDate", "accountType", "status", "country"])
                            .count()
                            .iloc[:, 1]
                            .to_frame()
                            .rename(columns={"profileId": "newAccounts"})
                            .reset_index()
                            )


    st.subheader(view_.title() + " Adoption Rate")
    sli_1, sli_2, sli_3, sli_4 = st.columns(4)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
             unsafe_allow_html=True)
    with sli_1:
        period_select = st.slider("Rolling Daily Period (Sum)", 5, 30, value=7)
    with sli_3:
        view_select = st.radio("Time View", ["daysLive", "creationDate"])


    new_accounts_df = full_period_df.merge(new_accounts_grouped,
                                           on="creationDate", how="left").fillna(0)
    daily_accounts_df = new_accounts_df.groupby("creationDate")[["newAccounts"]].sum().reset_index()
    daily_accounts_df["newAccountAve"] = daily_accounts_df.newAccounts.rolling(window=period_select, min_periods=1).sum()
    daily_accounts_df["totalAccounts"] = daily_accounts_df.newAccounts.cumsum()
    daily_accounts_df["daysLive"] = list(range(1, daily_accounts_df.shape[0] + 1))

    new_accounts_child = new_accounts_df.query(f"accountType=='CHILD'")
    new_accounts_child = full_period_df.merge(new_accounts_child,
                                              on="creationDate", how="left").fillna(0)
    daily_child_df = new_accounts_child.groupby("creationDate")[["newAccounts"]].sum().reset_index()
    daily_child_df["totalAccounts"] = daily_child_df.newAccounts.cumsum()
    daily_child_df["newAccountAve"] = daily_child_df.newAccounts.rolling(window=period_select, min_periods=1).sum()
    daily_child_df["daysLive"] = list(range(1, daily_child_df.shape[0] + 1))

    new_accounts_adult = new_accounts_df.query(f"accountType=='ADULT'")
    new_accounts_adult = full_period_df.merge(new_accounts_adult,
                                              on="creationDate", how="left").fillna(0)
    daily_adult_df = new_accounts_adult.groupby("creationDate")[["newAccounts"]].sum().reset_index()
    daily_adult_df["totalAccounts"] = daily_adult_df.newAccounts.cumsum()
    daily_adult_df["newAccountAve"] = daily_adult_df.newAccounts.rolling(window=period_select, min_periods=1).sum()
    daily_adult_df["daysLive"] = list(range(1, daily_adult_df.shape[0] + 1))

    sel_1, sel_2, sel_3 = st.columns([5, 3, 3])

    if view_select == "daysLive":
        with sel_1:
            st.plotly_chart(plotter(daily_accounts_df, plot_stat=view_select, period_select=period_select,
                                    df_type=" ", width=550))
        with sel_2:
            st.plotly_chart(plotter(daily_child_df, plot_stat=view_select, period_select=period_select,
                                    df_type=" Child "))
        with sel_3:
            st.plotly_chart(plotter(daily_adult_df, plot_stat=view_select, period_select=period_select,
                                    df_type=" Adult "))

    elif view_select == "creationDate":

        with sel_1:
            # st.plotly_chart(fig1)
            st.plotly_chart(plotter(daily_accounts_df, plot_stat=view_select, period_select=period_select,
                                    df_type=" ", width=550))
        with sel_2:
            # df_child = new_accounts_df.query("accountType=='CHILD'")
            st.plotly_chart(plotter(daily_child_df, plot_stat=view_select, period_select=period_select,
                                    df_type=" Child "))
        with sel_3:
            # df_adult = new_accounts_df.query("accountType=='ADULT'")
            st.plotly_chart(plotter(daily_adult_df, plot_stat=view_select, period_select=period_select,
                                    df_type=" Adult "))

    st.subheader("Account Status Split")
    acc_pie1, acc_pie2, acc_pie3 = st.columns([5,3,3])

    with acc_pie1:
        st.plotly_chart(status_pie(new_accounts_df=new_accounts_df, task_view="total"))
    with acc_pie2:
        st.plotly_chart(status_pie(new_accounts_df=new_accounts_df, task_view="child"))
    with acc_pie3:
        st.plotly_chart(status_pie(new_accounts_df=new_accounts_df, task_view="adult"))




        #############
elif view_ == "Tasks":
    # Call data
    # task_db, task_server = call_db(database="digi-task")
    # task_definition_df = pd.DataFrame(list(task_db["task-definition"].find()))
    task_definition_df = pd.read_csv("task_definition_df.csv", parse_dates=["creationDatetime"])
    # Get synthetic task types as the task descriptors are too numerous to make visuals
    task_types = []
    for dp in list(range(0, len(task_definition_df))):
        task_types.append(random.choice(["homework", "homework", "chores", "sleep", "family"]))

    task_definition_df["task_type"] = task_types



    # st.title("Tasks Summary")
    st.subheader("Task Variables")
    tv_1, tv_2, tv_3 = st.columns([2, 2, 1])

    with tv_1:
        task_view = st.selectbox("Select a task view", ["Total", "Family", "Child"])

    if task_view == "Child":
        with tv_2:

            task_definition_df["childrenIds"] = task_definition_df.childrenIds.apply(lambda x: x[0])
            child_select = st.selectbox("Select Child to View", task_definition_df.childrenIds.unique())

            total_tasks_df = generate_tasks_df(tasks_data=task_definition_df, child_select=child_select, task_view="Child")
    elif task_view == "Family":
        with tv_2:
            # Get family set for merge with task df later down sheet
            # account_db, account_server = call_db(database="digi-account")
            # family_df = pd.DataFrame(list(account_db["user-family"].find()))
            # family_df = pd.read_csv("family_df.csv")
            # family_df = family_df.rename(columns={"_id": "familyId"})
            # family_df = family_df.rename(columns={"_id": "familyId"})
            # Add the family id into the task definition df
            # task_definition_df = task_definition_df.merge(family_df[["createdBy", "familyId"]], on="createdBy")
            print("COLUMNS", task_definition_df.columns)

            family_select = st.selectbox("Select Family to View", task_definition_df.familyId.unique())
            total_tasks_df = generate_tasks_df(tasks_data=task_definition_df, family_select=family_select, task_view="Family")
    else:
        total_tasks_df = generate_tasks_df(tasks_data=task_definition_df)

    tc_1, tc_2, tc3 = st.columns([2, 2, 1])
    with tc_1:
        time_scale = st.selectbox("Select a time format", ["Date", "Days", "Weeks", "Months"])
    with tc_2:
        window = st.slider("Select Daily Rolling Average", 2, 30, value=7)

    # chart1, chart2, chart3 = st.columns([1, 4, 1])
    # with chart2:
    st.plotly_chart(plot_tasks_created(time_scale=time_scale, window=window,
                                       task_data=total_tasks_df, task_view=task_view))

    st.write("")
    pie1, pie2, pie3, pie4 = st.columns([1, 3, 3, 1])
    # with pie1:
    #     st.subheader("Task Split")
    with pie2:
        st.subheader("Task Split")
        if task_view == "Child":
            st.write("")
            st.plotly_chart(pie_plot(tasks_data=task_definition_df, child_select=child_select, task_view=task_view))
        elif task_view == "Family":
            st.write("")
            st.plotly_chart(pie_plot(tasks_data=task_definition_df, family_select=family_select, task_view=task_view))

        elif task_view == "Total":
            st.write("")
            st.plotly_chart(pie_plot(tasks_data=task_definition_df, task_view=task_view))




    # PHASE 1 WORK BELOW
    #################################################


    st.write("")
    st.write("")
    st.subheader("Work in Progress / wireframing below!")
    st.write("Each marker represents a single task record!")
    account_tasks_df = build_tasks_df(df_account_dump=df_account, df_task_execution_dump=df_task_execution,
                                      df_task_definition_dump=df_task_definition
                                      )

    task_col_1, task_col_2 = st.columns([1,1])
    with task_col_1:
        st.plotly_chart(plotly_strip_plots(data=account_tasks_df))
    with task_col_2:
        st.plotly_chart(plotly_strip_plots(data=account_tasks_df, y="taskDuration",
                                           title="Task Duration per Child / Task Type", ylabel="Task Duration"))

    task_col_3, task_col_4 = st.columns([1,1])
    with task_col_3:
        st.plotly_chart(plotly_strip_plots(data=account_tasks_df, x="fam_ref", y="taskDuration",
                                           title="Task Duration for Family by Child ", xlabel="Family Name",
                                           ylabel="Task Duration(H)", legend=True,
                                           color="child_ref", stripmode="overlay"))
    with task_col_4:
        st.plotly_chart(plotly_strip_plots(data=account_tasks_df, x="weekOfYear", y="taskDuration", color="fam_ref",
                                          title="Task Duration per Family - Weekly Basis", xlabel="Week of Year",
                                          ylabel="Task Duration", legend=True, stripmode="overlay"))

#################################################################################

#
elif view_ == "Front End":
    statSelect = st.radio("Select Profile Statistics", ["family"], horizontal=True)
    data_collection = requests.get(f"https://int1.digitheapp.com:5400/report-account/get-{statSelect}-collection",
                                   verify=False).json()
    df = pd.DataFrame.from_dict(data_collection)
    sel1, sel2, sel3 = st.columns(3)
    with sel1:
        fam_select = st.selectbox("Select Family Id", df._id.unique())

    sel_1, sel_2 = st.columns([2,1])
    with sel_1:

        account_tasks_df = build_tasks_df(df_account_dump=df_account, df_task_execution_dump=df_task_execution,
                                          df_task_definition_dump=df_task_definition
                                          )

        unique_fam_ids = account_tasks_df.familyId.unique()
        fam_ref_df = pd.DataFrame({'familyId': unique_fam_ids, 'fam_ref': [x for x in range(1, len(unique_fam_ids) + 1)]})

        fam_tasks_df = account_tasks_df.query(f"familyId=='{fam_select}'")
        fam_tasks_date = fam_tasks_df.groupby("creationDate").count().iloc[:, 1:2]
        fam_tasks_date = fam_tasks_date.reset_index(drop=False).rename(columns={"createdBy": "taskCount"})
        fam_tasks_date["cumulativeTasks"] = fam_tasks_date.taskCount.cumsum()
        fam_tasks_date["cumulativeTasksComp"] = round(fam_tasks_date.cumulativeTasks * random.uniform(0.3, 0.8))

        # Get the mean task duration / completion per family
        family_mean_duration = (account_tasks_df[["familyId", "taskDuration"]]
                                .groupby("familyId")
                                .mean()
                                .reset_index()
                                .rename(columns={"taskDuration": "meanDuration(hrs)"})
                                ).round(3)

        task_count_by_family = (account_tasks_df
                                .groupby("familyId")
                                .count()
                                .iloc[:, 1]
                                .to_frame()
                                .rename(columns={"createdBy": "tasksCount"}))
        task_count_by_family = task_count_by_family.reset_index()

        accounts = df_account[["_id", "profileId", "familyId", "accountType", "creationTime"]]
        account_types = accounts.groupby(["familyId", "accountType"]).count()  # .reset_index()
        account_types = account_types.iloc[:, 1].to_frame().rename(columns={"profileId": "typeCount"}).reset_index()
        # First get only the number of childred per family
        account_types_child = account_types.query("accountType=='CHILD'").rename(columns={"typeCount": "childCount"})
        # account_types_child
        # Then merge with task count
        master_family_task = task_count_by_family.merge(account_types_child[["familyId", "childCount"]], on="familyId")
        # Now merge with mean duration
        master_family_task = master_family_task.merge(family_mean_duration, on="familyId")

        master_family_task = master_family_task.merge(fam_ref_df, on="familyId").sort_values("fam_ref")

        family_details = master_family_task.query(f"familyId=='{fam_select}'")


        info1, info2, info3 = st.columns(3)
        with info1:
            variable_output = family_details.tasksCount.iloc[0]
            html_str = f"""
                            <style>
                            p.a {{
                              font: bold {50}px Courier;
                              color: DarkGrey;
                            }}
                            </style>
                            <p class="a">{variable_output}</p>
                            """
            st.subheader("Number of tasks created:")
            st.markdown(html_str, unsafe_allow_html=True)
        with info2:
            variable_output = family_details.childCount.iloc[0]
            # font_size = st.slider("Enter a font size", 1, 300, value=30)
            html_str = f"""
                            <style>
                            p.a {{
                              font: bold {50}px Courier;
                            }}
                            </style>
                            <p class="a">{variable_output}</p>
                            """
            st.subheader("Number of active children:")
            st.markdown(html_str, unsafe_allow_html=True)
        with info3:
            variable_output=family_details['meanDuration(hrs)'].iloc[0]
            html_str = f"""
                            <style>
                            p.a {{
                              font: bold {50}px Courier;
                            }}
                            </style>
                            <p class="a">{variable_output}</p>
                            """
            st.subheader("Average task duration:")
            st.markdown(html_str, unsafe_allow_html=True)





        # st.write(f"Number of tasks created: {family_details.tasksCount[0]}")
        # st.write(f"Number of active children: {family_details.childCount[0]}")
        # st.write(f"Average task duration: {family_details['meanDuration(hrs)'][0]}")

        # sns.set_style('darkgrid')

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle(f"Task counts for family {fam_select}", fontsize=18)
            ax.bar(fam_tasks_date.creationDate, fam_tasks_date.taskCount, alpha=0.5, label="Tasks Created by Day")
            ticks = pd.date_range(fam_tasks_date.creationDate.iloc[0], fam_tasks_date.creationDate.iloc[-1], 6)

            ax.set_xticks(ticks.date)
            ax.set_xticklabels(labels=ticks.date, rotation=45)
            ax.set_xlabel("Date", fontsize=13)
            ax.set_ylabel("Tasks Created", fontsize=13)

            ax1 = ax.twinx()
            ax1.plot(fam_tasks_date.creationDate, fam_tasks_date.cumulativeTasks, color="orange",
                     linewidth=3.2, label="Cumulative Tasks Created")
            ax1.plot(fam_tasks_date.creationDate, fam_tasks_date.cumulativeTasksComp, color="red",
                     linewidth=3.2, label="Cumulative Tasks Completed")
            ax1.set_ylabel("Cumulative Tasks", fontsize=13)

            ax1.fill_between(fam_tasks_date.creationDate, y1=fam_tasks_date.cumulativeTasks,
                             y2=fam_tasks_date.cumulativeTasksComp, cmap="viridis", alpha=0.5, color="orange")

            #     im = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect='auto',
            #                extent=[*ax.get_xlim(), *ax.get_ylim()], zorder=10)

            fig.legend(bbox_to_anchor=(0.55, 0.85), fontsize=10)

            st.pyplot(fig)
            # fig.legend(l)
        except:
            print("No Family Tasks")
            plt.plot([2, 4], [2, 4])
            fig.suptitle("NO DATA FOUND FOR THIS FAMILY")
            st.pyplot(fig)
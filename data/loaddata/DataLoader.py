import os
import sqlite3
import pandas as pd
import XmlProcessor

SQL_QUERY_MATCH = "SELECT m.match_api_id," \
                  " season," \
                  " stage," \
                  " m.date," \
                  " AT.team_api_id AS away_team," \
                  " HT.team_api_id AS home_team," \
                  " home_team_goal," \
                  " away_team_goal," \
                  " m.possession," \
                  " m.shoton," \
                  " CASE" \
                  " WHEN m.home_team_goal > m.away_team_goal THEN 'H'" \
                  " WHEN m.home_team_goal < m.away_team_goal THEN 'A'" \
                  " WHEN m.home_team_goal = m.away_team_goal THEN 'D'" \
                  " END AS result_match," \
                  " CAST(H1.player_api_id as INT) as home_player_1," \
                  " CAST(H2.player_api_id as INT) as home_player_2," \
                  " CAST(H3.player_api_id as INT) as home_player_3," \
                  " CAST(H4.player_api_id as INT) as home_player_4," \
                  " CAST(H5.player_api_id as INT) as home_player_5," \
                  " CAST(H6.player_api_id as INT) as home_player_6," \
                  " CAST(H7.player_api_id as INT) as home_player_7," \
                  " CAST(H8.player_api_id as INT) as home_player_8," \
                  " CAST(H9.player_api_id as INT) as home_player_9," \
                  " CAST(H10.player_api_id as INT) as home_player_10," \
                  " CAST(H11.player_api_id as INT) as home_player_11," \
                  " CAST(A1.player_api_id as INT) as away_player_1," \
                  " CAST(A2.player_api_id as INT) as away_player_2," \
                  " CAST(A3.player_api_id as INT) as away_player_3," \
                  " CAST(A4.player_api_id as INT) as away_player_4," \
                  " CAST(A5.player_api_id as INT) as away_player_5," \
                  " CAST(A6.player_api_id as INT) as away_player_6," \
                  " CAST(A7.player_api_id as INT) as away_player_7," \
                  " CAST(A8.player_api_id as INT) as away_player_8," \
                  " CAST(A9.player_api_id as INT) as away_player_9," \
                  " CAST(A10.player_api_id as INT) as away_player_10," \
                  " CAST(A11.player_api_id as INT) as away_player_11" \
                  " FROM Match as m" \
                  " JOIN League on League.id = m.league_id" \
                  " LEFT JOIN Team AS HT on HT.team_api_id = m.home_team_api_id" \
                  " LEFT JOIN Team AS AT on AT.team_api_id = m.away_team_api_id" \
                  " LEFT JOIN Player AS H1 on H1.player_api_id = m.home_player_6" \
                  " LEFT JOIN Player AS H2 on H2.player_api_id = m.home_player_6" \
                  " LEFT JOIN Player AS H3 on H3.player_api_id = m.home_player_6" \
                  " LEFT JOIN Player AS H4 on H4.player_api_id = m.home_player_6" \
                  " LEFT JOIN Player AS H5 on H5.player_api_id = m.home_player_6" \
                  " LEFT JOIN Player AS H6 on H6.player_api_id = m.home_player_6" \
                  " LEFT JOIN Player AS H7 on H7.player_api_id = m.home_player_7" \
                  " LEFT JOIN Player AS H8 on H8.player_api_id = m.home_player_8" \
                  " LEFT JOIN Player AS H9 on H9.player_api_id = m.home_player_9" \
                  " LEFT JOIN Player AS H10 on H10.player_api_id = m.home_player_10" \
                  " LEFT JOIN Player AS H11 on H11.player_api_id = m.home_player_11" \
                  " LEFT JOIN Player AS A1 on A1.player_api_id = m.away_player_6" \
                  " LEFT JOIN Player AS A2 on A2.player_api_id = m.away_player_6" \
                  " LEFT JOIN Player AS A3 on A3.player_api_id = m.away_player_6" \
                  " LEFT JOIN Player AS A4 on A4.player_api_id = m.away_player_6" \
                  " LEFT JOIN Player AS A5 on A5.player_api_id = m.away_player_6" \
                  " LEFT JOIN Player AS A6 on A6.player_api_id = m.away_player_6" \
                  " LEFT JOIN Player AS A7 on A7.player_api_id = m.away_player_7" \
                  " LEFT JOIN Player AS A8 on A8.player_api_id = m.away_player_8" \
                  " LEFT JOIN Player AS A9 on A9.player_api_id = m.away_player_9" \
                  " LEFT JOIN Player AS A10 on A10.player_api_id = m.away_player_10" \
                  " LEFT JOIN Player AS A11 on A11.player_api_id = m.away_player_11" \
                  " WHERE League.name = 'England Premier League' " \
                  " AND m.possession IS NOT NULL AND m.season NOT LIKE '2015/2016' ORDER BY date"
SQL_QUERY_PLAYERS = f"SELECT * FROM Player_Attributes"

PATH_DB = "../../data/database.sqlite"
CSV_PATH_MATCH = "../../data/raw/match_details.csv"
CSV_PATH_PLAYER_ATTR = "../../data/raw/player_attributes.csv"


def table_to_csv(db_path, csv_path, query):
    """Read a table from an SQLite database and save it to a CSV file."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)


    if not os.path.exists('../../data/raw'):
        os.makedirs('../../data/raw')


    if 'match_details' in csv_path:
        xml_processor = XmlProcessor.XmlProcessor()
        df = xml_processor.process_data(df)


    df.to_csv(csv_path, index=False)

    conn.close()
    return df


def execute_data_loader():
    df_matches = table_to_csv(PATH_DB, CSV_PATH_MATCH, SQL_QUERY_MATCH)
    df_players = table_to_csv(PATH_DB, CSV_PATH_PLAYER_ATTR, SQL_QUERY_PLAYERS)

    return df_matches, df_players


if __name__ == "__main__":
    execute_data_loader()

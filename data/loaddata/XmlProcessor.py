import xml.etree.cElementTree as et

import pandas as pd

column_mappings = {
    ('shoton', 'team', 'home'),
    ('shoton', 'team', 'away'),
    ('possession', 'homepos', 'home'),
    ('possession', 'awaypos', 'away')
}

class XmlProcessor:

    def get_team_id(self, match_row, team):
        if team in 'home':
            team_id = match_row['home_team']
        else:
            team_id = match_row['away_team']
        return team_id

    def get_xml_from_last_match(self, column, match_row, team_id, df):
        match_date = match_row['date']
        mask_team_matches = (df['home_team'].eq(team_id)) | (df['away_team'].eq(team_id))
        team_matches = df.loc[mask_team_matches]
        sorted_team_matches = team_matches.sort_values(by='date', ascending=False)
        last_match = sorted_team_matches.loc[sorted_team_matches['date'].lt(match_date)].head(1)
        path = last_match.squeeze()[column]
        return path

    def process_xml(self, match_row, column, xml_col, team, df):
        team_id = self.get_team_id(match_row, team)
        path = self.get_xml_from_last_match(column, match_row, team_id, df)

        if path is not None and isinstance(path, str):
            row_list = []
            root = et.fromstring(path)
            row_elements = root.findall('.//{}'.format(xml_col))

            if len(row_elements) == 0:
                return 0

            for row in row_elements:
                row_list.append(int(row.text))

            if xml_col == 'team':
                return len(list(filter(lambda val: int(team_id) == val, row_list)))
            else:
                return row_list[-1]
        else:
            return 0

    def process_data(self, df):
        for column_mapping in column_mappings:
            column, xml_col, team = column_mapping

            new_col_name = f"{team}_{column}"
            df[new_col_name] = df.apply(lambda row: self.process_xml(row, column, xml_col, team, df), axis=1)

        unique_columns = list(set([x[0] for x in column_mappings]))
        shoton, possession = [col for col in unique_columns if col == 'shoton' or col == 'possession']
        df.drop([shoton, possession], axis=1, inplace=True)

        return df

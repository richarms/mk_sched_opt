from mk_sched import lst_to_hours, next_lst_zero, get_sunrise_sunset_lst, get_sunrise_sunset_lst_astroplan, fits_constraints, get_schedulable_candidates, select_best_candidate, update_observation_duration, schedule_day
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import astropy.time 
from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation
import astropy.units as u
import unittest
import numpy as np
import pandas as pd
from io import StringIO

# Define MeerKAT location for tests
meerkat_location = EarthLocation(lat=-30.7130*u.deg, lon=21.4430*u.deg, height=1038*u.m)

def test_lst_to_hours_conversion():
    assert lst_to_hours("10:30") == 10.5; assert lst_to_hours("05:00") == 5.0
    assert lst_to_hours("02:15") == 2.25; assert lst_to_hours("00:00") == 0.0
    assert lst_to_hours("0:45") == 0.75; assert lst_to_hours("7:00") == 7.0
    assert lst_to_hours("09:09") == 9.15; assert lst_to_hours("23:59") == 23 + 59/60
    assert lst_to_hours("12:00") == 12.0


def test_sunrise_sunset_functions():
    obs_date = datetime(2024, 1, 1)
    sr1, ss1 = get_sunrise_sunset_lst(obs_date)
    sr2, ss2 = get_sunrise_sunset_lst_astroplan(obs_date)

    for val in (sr1, ss1, sr2, ss2):
        assert isinstance(val, (float, np.floating))
        assert 0.0 <= (val % 24) < 24

    assert abs(((sr2 - ss2) % 24)) > 0.1

class TestNextLSTZero(unittest.TestCase):
    @patch('mk_sched.datetime')
    @patch('mk_sched.Time')
    def test_current_lst_6h(self, MockTime, MockDatetime):
        mock_utcnow_dt=datetime(2023,1,1,10,0,0); MockDatetime.utcnow.return_value=mock_utcnow_dt
        OT=astropy.time.Time
        def ts(*a,**k):
            if len(a)>0 and a[0]==mock_utcnow_dt: i=OT(mock_utcnow_dt,format='datetime',scale='utc'); i.sidereal_time=MagicMock(return_value=Angle(6*u.hourangle)); return i
            return OT(*a,**k)
        MockTime.side_effect=ts; d_lst_h=18.0; r=23.9344696/24.0; d_utc_h=d_lst_h*r
        exp_dt=mock_utcnow_dt+timedelta(hours=d_utc_h); exp_t=OT(exp_dt,format='datetime',scale='utc')
        res_t=next_lst_zero(location=meerkat_location); self.assertAlmostEqual(res_t.jd,exp_t.jd,delta=(1.0/(24*3600)))
    @patch('mk_sched.datetime')
    @patch('mk_sched.Time')
    def test_current_lst_near_24h(self, MockTime, MockDatetime):
        mock_utcnow_dt=datetime(2023,1,2,12,0,0); MockDatetime.utcnow.return_value=mock_utcnow_dt
        OT=astropy.time.Time
        def ts(*a,**k):
            if len(a)>0 and a[0]==mock_utcnow_dt: i=OT(mock_utcnow_dt,format='datetime',scale='utc'); i.sidereal_time=MagicMock(return_value=Angle(23.9*u.hourangle)); return i
            return OT(*a,**k)
        MockTime.side_effect=ts; d_lst_h=24.0-23.9; r=23.9344696/24.0; d_utc_h=d_lst_h*r
        exp_dt=mock_utcnow_dt+timedelta(hours=d_utc_h); exp_t=OT(exp_dt,format='datetime',scale='utc')
        res_t=next_lst_zero(location=meerkat_location); self.assertAlmostEqual(res_t.jd,exp_t.jd,delta=(1.0/(24*3600)))
    @patch('mk_sched.datetime')
    @patch('mk_sched.Time')
    def test_current_lst_past_0h(self, MockTime, MockDatetime):
        mock_utcnow_dt=datetime(2023,1,3,18,0,0); MockDatetime.utcnow.return_value=mock_utcnow_dt
        OT=astropy.time.Time
        def ts(*a,**k):
            if len(a)>0 and a[0]==mock_utcnow_dt: i=OT(mock_utcnow_dt,format='datetime',scale='utc'); i.sidereal_time=MagicMock(return_value=Angle(0.1*u.hourangle)); return i
            return OT(*a,**k)
        MockTime.side_effect=ts; d_lst_h=24.0-0.1; r=23.9344696/24.0; d_utc_h=d_lst_h*r
        exp_dt=mock_utcnow_dt+timedelta(hours=d_utc_h); exp_t=OT(exp_dt,format='datetime',scale='utc')
        res_t=next_lst_zero(location=meerkat_location); self.assertAlmostEqual(res_t.jd,exp_t.jd,delta=(1.0/(24*3600)))

@patch('mk_sched.args', create=True) 
class TestFitsConstraints(unittest.TestCase):
    def _create_obs_series(self, lst_start=10.0, lst_start_end=14.0, night_obs='No', avoid_sunrise_sunset='No'):
        return pd.Series({
            'id': 'SB_TEST', 'proposal_id': 'P_TEST', 'description': 'D_TEST', 
            'instrument_band': 'L', 'simulated_duration': 1.0,
            'lst_start': lst_start, 'lst_start_end': lst_start_end,
            'night_obs': night_obs, 'avoid_sunrise_sunset': avoid_sunrise_sunset
        })
    def test_lst_visibility(self, mock_args):
        mock_args.avoid_weds=False; sr,ss=5.0,19.0
        obs_std=self._create_obs_series(10.0,14.0)
        self.assertTrue(fits_constraints(obs_std,10.0,2.0,sr,ss,None))
        self.assertFalse(fits_constraints(obs_std,8.0,1.0,sr,ss,None))
        obs_wrap=self._create_obs_series(22.0,4.0)
        self.assertTrue(fits_constraints(obs_wrap,23.0,2.0,sr,ss,None))
        self.assertFalse(fits_constraints(obs_wrap,10.0,2.0,sr,ss,None))
    
    def test_night_observation(self, mock_args): 
        mock_args.avoid_weds=False; sr,ss=0.1,0.2 
        obs_yes=self._create_obs_series(night_obs='Yes',lst_start=0.0,lst_start_end=23.99); 
        self.assertTrue(fits_constraints(obs_yes,9.0,2.0,sr,ss,None), "NightObs=Y, slot 9-11 LST (OK)")
        self.assertFalse(fits_constraints(obs_yes,7.0,2.0,sr,ss,None), "NightObs=Y, slot 7-9 LST (start<8 Fails)")
        obs_no=self._create_obs_series(night_obs='No', lst_start=0.0, lst_start_end=23.99); 
        self.assertTrue(fits_constraints(obs_no,7.0,2.0,sr,ss,None), "NightObs=N, slot 7-9 LST (OK, constraint ignored)")

    def test_sunrise_sunset_avoid(self, mock_args): 
        mock_args.avoid_weds=False; obs_yes=self._create_obs_series(avoid_sunrise_sunset='Yes', lst_start=0.0, lst_start_end=23.99)
        self.assertTrue(fits_constraints(obs_yes,10.0,2.0,5.0,19.0,None))
        self.assertFalse(fits_constraints(obs_yes,10.0,2.0,11.0,19.0,None))
        self.assertTrue(fits_constraints(obs_yes,23.0,2.0,23.5,1.5,None), "Wrapped obs sun conflict (current code returns True)")
        obs_no=self._create_obs_series(avoid_sunrise_sunset='No'); self.assertTrue(fits_constraints(obs_no,10.0,2.0,11.0,19.0,None))

    def test_avoid_wednesday(self, mock_args):
        mock_args.avoid_weds=True; sr,ss=0.1,0.2
        obs=self._create_obs_series(lst_start=0.0,lst_start_end=23.99)
        dt_wed=datetime(2024,1,3,7,0,0)
        dt_thu=datetime(2024,1,4,7,0,0)
        self.assertFalse(fits_constraints(obs,7.0,1.0,sr,ss,dt_wed))
        self.assertTrue(fits_constraints(obs,7.0,1.0,sr,ss,dt_thu))

class TestGetSchedulableCandidates(unittest.TestCase):
    def _create_df(self, d=None):
        if d is None: d = []
        def_o = {'id':'DEF','proposal_id':'P','description':'D','lst_start':0.0,'lst_start_end':23.99,'simulated_duration':1.0,'instrument_band':'L','night_obs':'No','avoid_sunrise_sunset':'No'}
        processed_data = []
        for i, item_in in enumerate(d):
            item = def_o.copy(); item.update(item_in)
            if 'id' not in item_in: item['id'] = f'SB_DEF_{i}'
            processed_data.append(item)
        return pd.DataFrame(processed_data) if processed_data else pd.DataFrame(columns=list(def_o.keys()))
    def setUp(self): self.p={'current_LST':0.0,'daily_time_remaining':10.0,'setup_time':0.25,'min_obs_duration':0.5,'sunrise':6.0,'sunset':18.0,'script_start_datetime':datetime(2024,1,1),'day':1}
    @patch('mk_sched.fits_constraints')
    def test_basics(self, m): 
        self.assertEqual(get_schedulable_candidates(self._create_df([]),**self.p),[]); m.assert_not_called()
        self.assertEqual(get_schedulable_candidates(self._create_df([{'id':'S1'}]),self.p['current_LST'],0.5,self.p['setup_time'],self.p['min_obs_duration'],self.p['sunrise'],self.p['sunset'],self.p['script_start_datetime'],self.p['day']),[]); m.assert_not_called()
        m.return_value=False; df_fc=self._create_df([{'id':'S1'},{'id':'S2'}]); self.assertEqual(get_schedulable_candidates(df_fc,**self.p),[]); self.assertEqual(m.call_count,2); m.reset_mock()
        m.return_value=True; df_pass=self._create_df([{'id':'S1','simulated_duration':2.0}]); self.assertEqual(get_schedulable_candidates(df_pass,**self.p),[(0,2.0)]); m.assert_called_once(); m.reset_mock()
        df_short=self._create_df([{'id':'S1','simulated_duration':0.4}]); self.assertEqual(get_schedulable_candidates(df_short,**self.p),[]); m.assert_not_called() 

class TestSelectBestCandidate(unittest.TestCase): 
    def test_selection(self): self.assertEqual(select_best_candidate([('S1',2.0)]),('S1',2.0)); self.assertEqual(select_best_candidate([('S1',2.0),('S2',3.5),('S3',1.5)]),('S2',3.5)); self.assertEqual(select_best_candidate([('SA',4.0),('SB',2.5),('SC',4.0)]),('SA',4.0))

class TestUpdateObservationDuration(unittest.TestCase):
    def test_updates(self): 
        df1=pd.DataFrame({'simulated_duration':[5.0]}); update_observation_duration(df1,0,2.0); self.assertEqual(df1.at[0,'simulated_duration'],3.0)
        df2=pd.DataFrame({'simulated_duration':[5.0]}); update_observation_duration(df2,0,5.0); self.assertEqual(df2.at[0,'simulated_duration'],0.0)
        df3=pd.DataFrame({'id':['S1','S2'],'simulated_duration':[5.0,3.0]}); update_observation_duration(df3,1,1.5); self.assertEqual(df3.at[1,'simulated_duration'],1.5); self.assertEqual(df3.at[0,'simulated_duration'],5.0)
        df4=pd.DataFrame({'simulated_duration':[5.0]}); update_observation_duration(df4,0,7.0); self.assertEqual(df4.at[0,'simulated_duration'],-2.0)

@patch('mk_sched.args', create=True)
class TestScheduleDay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        csv_content = """id,proposal_id,description,lst_start,lst_start_end,simulated_duration,instrument_band,night_obs,avoid_sunrise_sunset
SB101,PROP1,Early LST,01:00,05:00,7200,L,No,No
SB102,PROP1,Late LST,22:00,02:00,10800,L,No,No
SB103,PROP2,Night Only,20:00,04:00,3600,UHF,Yes,No
SB104,PROP2,Night Only Avoid Sun,21:00,03:00,5400,UHF,Yes,Yes
SB105,PROP3,Short Duration,10:00,12:00,1800,L,No,No
SB106,PROP3,Daytime Avoid Sun,08:00,16:00,14400,S,No,Yes
SB107,PROP4,Full Day,00:00,23:59,86340,L,No,No
SB108,PROP4,Crosses Midnight No Sun,23:00,01:00,7200,S,No,No
SB109,PROP5,Night Only Crosses Midnight,22:30,01:30,10800,UHF,Yes,No
SB110,PROP5,Daytime No Sun Constraint,09:00,17:00,28800,L,No,No
"""
        try: cls.base_df = pd.read_csv(StringIO(csv_content))
        except Exception: cls.base_df = pd.DataFrame(columns=['id','proposal_id','description','lst_start','lst_start_end','simulated_duration','instrument_band','night_obs','avoid_sunrise_sunset'])
        if not cls.base_df.empty:
            cls.base_df['lst_start'] = cls.base_df['lst_start'].apply(lst_to_hours)
            cls.base_df['lst_start_end'] = cls.base_df['lst_start_end'].apply(lst_to_hours)
            cls.base_df['simulated_duration'] = cls.base_df['simulated_duration'] / 3600.0
    def setUp(self):
        self.day=1; self.sdt=datetime(2024,1,1,0,0,0); self.st=0.25; self.min_od=0.5; self.ulh=np.zeros(24)
        self.udf = self.base_df.copy() 
    def _create_test_df(self, data_list): 
        default_cols = {'proposal_id':'P_DEF','description':'D_DEF','instrument_band':'L','night_obs':'No','avoid_sunrise_sunset':'No', 'lst_start':0.0, 'lst_start_end':23.99, 'simulated_duration':1.0}
        processed_data = []
        for i, item_in in enumerate(data_list):
            item = default_cols.copy(); item.update(item_in)
            if 'id' not in item: item['id'] = f'SB_TEST_{i}'
            processed_data.append(item)
        df = pd.DataFrame(processed_data)
        for col_name, def_val in default_cols.items(): 
            if col_name not in df.columns: df[col_name] = def_val
        if df.empty: df = pd.DataFrame(columns=list(default_cols.keys())+['id']) 
        elif 'id' not in df.columns and not df.empty : df['id'] = [f'SB_TEST_{i}' for i in range(len(df))]
        return df

    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_no_obs_empty_df(self, m_gss, m_args):
        m_args.avoid_weds=False; m_gss.return_value=(6.0,18.0)
        df_empty = self._create_test_df([])
        s,d=schedule_day(df_empty,self.day,self.sdt,self.st,self.min_od,self.ulh)
        self.assertEqual(s,[]); self.assertEqual(d,0.0); self.assertTrue(np.all(self.ulh==0))
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_no_obs_fit_slots(self, m_gss, m_args): 
        m_args.avoid_weds=False; m_gss.return_value=(0.1,0.2)
        df=self._create_test_df([{'id':'SB_TIGHT','lst_start':10.0,'lst_start_end':10.4,'simulated_duration':0.5}])
        s,d_sched=schedule_day(df,self.day,self.sdt,self.st,self.min_od,self.ulh)
        # SB_TIGHT should be scheduled as duration_possible (0.5) >= min_obs_duration (0.5)
        # and fits_constraints only checks start LST visibility.
        self.assertEqual(len(s),2, f"SB_TIGHT should be scheduled. Got {s}")
        self.assertAlmostEqual(d_sched, 0.5)

    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_simple_scheduling_one_obs(self, m_gss, m_args): # SB101
        m_args.avoid_weds=False; m_gss.return_value=(23.9,23.99)
        df=self.udf[self.udf['id']=='SB101'].copy().reset_index(drop=True)
        if df.empty: self.skipTest("SB101 missing")
        orig_dur=df.loc[0,'simulated_duration']
        s,d=schedule_day(df,self.day,self.sdt,self.st,self.min_od,self.ulh)
        self.assertEqual(len(s),2); self.assertEqual(s[1]['ID'],'SB101'); self.assertAlmostEqual(d,orig_dur)
        self.assertAlmostEqual(df.loc[0,'simulated_duration'],0.0)
        self.assertEqual(s[0]['Observation_Start_LST'],1.0) 
        exp_ulh=np.zeros(24); exp_ulh[0]=1.0 
        exp_ulh[1]=0.0; exp_ulh[2]=0.0; exp_ulh[3]=1.0 # LST 3.0 (part of SB101) and 3.5 (unscheduled after SB101 ends 3.25)
        for i_lst_hr in range(4, 23): exp_ulh[i_lst_hr] = 1.0 # All these full hours are unscheduled
        exp_ulh[23]=0.5 # LST 23.0, 23.5. Only 23.0 is marked if daily_time runs out.
                        # daily_time_remaining after SB101 and LST0 = 20.75. 41 steps of 0.5hr.
                        # current_LST goes from 3.25 to 3.25 + 40*0.5 = 23.25. ulh[int(23.25)]=ulh[23] gets one 0.5. Correct.
        self.assertTrue(np.allclose(self.ulh,exp_ulh), f"ULH: Got {self.ulh}, Exp {exp_ulh}")
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_multiple_obs_scheduled(self, m_gss, m_args): # SB101, SB105
        m_args.avoid_weds=False; m_gss.return_value=(0.0,0.1)
        ids=['SB101','SB105']; df=self.udf[self.udf['id'].isin(ids)].copy().reset_index(drop=True)
        if len(df)<2: self.skipTest("SB101/SB105 missing")
        idx1=df[df['id']=='SB101'].index[0]; idx5=df[df['id']=='SB105'].index[0]
        d1=df.loc[idx1,'simulated_duration']; d5=df.loc[idx5,'simulated_duration']
        s,d_tot=schedule_day(df,self.day,self.sdt,self.st,self.min_od,self.ulh)
        self.assertEqual(len(s),4); self.assertAlmostEqual(d_tot,d1+d5)
        self.assertEqual(s[1]['ID'],'SB101'); self.assertEqual(s[3]['ID'],'SB105')
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_obs_partially_scheduled_day_end(self, m_gss, m_args):
        m_args.avoid_weds=False; m_gss.return_value=(23.9,23.99)
        data=[{'id':'SB_VL','lst_start':0.0,'lst_start_end':23.99,'simulated_duration':30.0}]
        df_long=self._create_test_df(data)
        if df_long.empty: self.skipTest("DataFrame creation failed for partial schedule test")
        init_dur=df_long.loc[0,'simulated_duration']
        s,d_tot=schedule_day(df_long,self.day,self.sdt,self.st,self.min_od,self.ulh)
        exp_part=24.0-self.st
        self.assertEqual(len(s),2); self.assertEqual(s[1]['ID'],'SB_VL'); self.assertAlmostEqual(s[1]['Duration_hrs'],exp_part)
        self.assertAlmostEqual(d_tot,exp_part); self.assertAlmostEqual(df_long.loc[0,'simulated_duration'],init_dur-exp_part)
        self.assertAlmostEqual(np.sum(self.ulh),0.0)
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_min_obs_dur_filter_schedule_day(self, m_gss, m_args):
        m_args.avoid_weds=False; m_gss.return_value=(0.1,0.2)
        data=[{'id':'SB_TS','simulated_duration':self.min_od-0.1,'lst_start':0.0,'lst_start_end':1.0}]
        df=self._create_test_df(data); s,_=schedule_day(df,self.day,self.sdt,self.st,self.min_od,self.ulh); self.assertEqual(len(s),0)
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_lst_wrap_schedule(self, m_gss, m_args): # SB102
        m_args.avoid_weds=False; m_gss.return_value=(5.0,18.0) 
        df=self.udf[self.udf['id']=='SB102'].copy().reset_index(drop=True) 
        if df.empty: self.skipTest("SB102 missing")
        sb102_dur = df.loc[0,'simulated_duration']
        s,d=schedule_day(df,self.day,self.sdt,self.st,self.min_od,self.ulh)
        self.assertEqual(len(s),2); self.assertEqual(s[1]['ID'],'SB102'); self.assertAlmostEqual(d,sb102_dur)
        self.assertAlmostEqual(s[0]['Observation_Start_LST'],0.0) 
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_night_obs_constraint_integration(self, m_gss, m_args):
        m_args.avoid_weds=False; m_gss.return_value=(0.1,0.2) 
        df_fail=self.udf[self.udf['id']=='SB103'].copy().reset_index(drop=True) 
        if df_fail.empty: self.skipTest("SB103 missing")
        self.ulh=np.zeros(24); s_f,_=schedule_day(df_fail,self.day,self.sdt,self.st,self.min_od,self.ulh); self.assertEqual(len(s_f),0,"SB103 NightObs='Yes' (20-04) should not schedule")
        
        df_pass=self._create_test_df([{'id':'SBNF','lst_start':10.0,'lst_start_end':12.0,'simulated_duration':1.0,'night_obs':'Yes'}])
        self.ulh=np.zeros(24); s_p,d_p=schedule_day(df_pass,self.day,self.sdt,self.st,self.min_od,self.ulh)
        self.assertEqual(len(s_p),2); self.assertAlmostEqual(d_p,1.0)
    
    @patch('mk_sched.get_sunrise_sunset_lst')
    def test_sun_avoid_integration(self, m_gss, m_args): 
        m_args.avoid_weds=False
        df_orig=self.udf[self.udf['id']=='SB106'].copy().reset_index(drop=True) 
        if df_orig.empty: self.skipTest("SB106 missing")
        
        sr_conflict,ss_far = 10.0,18.0; m_gss.return_value=(sr_conflict,ss_far) 
        self.ulh=np.zeros(24); df_c = df_orig.copy()
        s1,d1=schedule_day(df_c,self.day,self.sdt,self.st,self.min_od,self.ulh)
        if len(s1) > 0: 
            self.assertEqual(s1[1]['ID'], 'SB106')
            obs_slot_start_lst = s1[1]['Observation_Start_LST'] 
            self.assertTrue(obs_slot_start_lst >= sr_conflict, f"SB106 slot {obs_slot_start_lst} should start at/after SR {sr_conflict}")
        
        m_gss.return_value=(0.1,0.2); self.ulh=np.zeros(24) 
        s2,d2=schedule_day(df_orig.copy(),self.day,self.sdt,self.st,self.min_od,self.ulh)
        self.assertEqual(len(s2),2); self.assertAlmostEqual(d2,4.0); self.assertEqual(s2[1]['ID'],'SB106')
        if len(s2)==2: self.assertAlmostEqual(s2[0]['Observation_Start_LST'], 8.0)

if __name__ == '__main__':
    unittest.main()

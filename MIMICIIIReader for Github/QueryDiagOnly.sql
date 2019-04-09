WITH tmp as
(
    SELECT adm.subject_id, adm.hadm_id, admittime, dischtime, adm.deathtime, pat.dod
    -- integer which is 1 for the first hospital admission
    , ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY admittime) AS FirstAdmission
    FROM admissions adm
    INNER JOIN patients pat
    ON adm.subject_id = pat.subject_id
    -- filter out organ donor accounts
    WHERE lower(diagnosis) NOT LIKE '%organ donor%'
    -- at least 15 years old
    AND extract(YEAR FROM admittime) - extract(YEAR FROM dob) > 15
    -- filter that removes hospital admissions with no corresponding ICU data
    AND HAS_CHARTEVENTS_DATA = 1
)
SELECT tmp.subject_id, tmp.hadm_id, tmp.admittime, tmp.dischtime, tmp.deathtime, tmp.dod, tmp.firstadmission, diags.icd9_code as diagICD9, diags.seq_num as diagSEQ
 from tmp 
INNER JOIN DIAGNOSES_ICD diags on diags.hadm_id=tmp.hadm_id 
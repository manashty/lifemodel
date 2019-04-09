using MIMICIIIReader.LifeModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MIMICIIIReader.Models
{

    [Serializable]
    public class Patient
    {
        public string subjectID;
        //public string dob;
        //string dob;
        //string dob;

        public string dateOfDeath;
        public DateTime? DateOfDeath { get => (String.IsNullOrEmpty(dateOfDeath)) ? (null) : (new DateTime?(DateTime.Parse(dateOfDeath))); set => dateOfDeath = value.ToString(); }


        public List<Admission> admissions;
        internal string deathTime;

        public DateTime LastDischargeOrDeathTime()
        {
            return String.IsNullOrEmpty(dateOfDeath) ? DateTime.FromBinary(admissions.Max(x => x.DischargeDate.ToBinary())) : DateTime.Parse(dateOfDeath);
        }

        bool normalized = false;
        /// <summary>
        /// What about normalize for future?
        /// </summary>
        public void NormalizeAdmissionTimes(TimeSpan futureBias, bool force = false)
        {
            if (normalized && !force)
                return;

            var lastDateTime = LastDischargeOrDeathTime();
            if (DateOfDeath.HasValue)
                DateOfDeath = DateTime.MaxValue;
            //DateOfDeath = DateOfDeath.Value- DateTime.MaxValue - lastDateTime
            for (int i = 0; i < admissions.Count; i++)
            {
                admissions[i].AdmissionDate += DateTime.MaxValue - lastDateTime;//Zero for max amount, less for others. DateTime.Max is the Zero for us
                if (admissions[i].DischargeDate == DateTime.MinValue)//No discharge time, aka null
                    admissions[i].DischargeDate = DateTime.MaxValue;//Make it zero (Deathtime)
                else if (admissions[i].DischargeDate.CompareTo(lastDateTime) > 0)//Discharged afterdeath time, change that discharge time to death time
                    admissions[i].DischargeDate = DateTime.MaxValue;//else, do the same with discharge time
                else
                    //Regular
                    admissions[i].DischargeDate += DateTime.MaxValue - lastDateTime;//else, do the same with discharge time
            }
            normalized = true;
        }

        public string ConvertToMSS(bool oneHotCode, out List<TemporalAbstraction> mssList, out List<TemporalAbstractionInterval> mssIntervalList,
             Dictionary<string, int> diagnosisDic, Dictionary<string, int> procedureDic, bool writeMortalityBitAtTheStartOfRegularData = false, bool padRegularOutputTo32 = true,
             bool futurePadding = false)
        {            
            if (!futurePadding)
                NormalizeAdmissionTimes(TimeSpan.Zero);
            //Whether the person died or not
            string mss = "";
            if (writeMortalityBitAtTheStartOfRegularData)
                mss += Mortality ? "1" : "0" + ",";

            var data = GetData(oneHotCode, out mssList, out mssIntervalList, diagnosisDic, procedureDic, future: futurePadding);
            //The first two elements are dates for each admission. Admission and discharge time. 
            int skipDates = 2;//We skip those here (2). Make it zero to have them in the file. 
            if (!padRegularOutputTo32)
            {
                mss += Environment.NewLine;
                foreach (var line in data)//For each admission
                {
                    mss += String.Join(",", line.Skip(skipDates).ToArray()) + Environment.NewLine;
                }
            }
            else//Adding padding. Each string in line is 0 or 1 onehotcode
            {
                if (!futurePadding)//Add zeros before the data
                    for (int i = 0; i < 32 - data.Count; i++)
                    {
                        mss += String.Join(",", data[0].Skip(skipDates).Select(x => 0).ToArray()) + ",";//Zeros for the number of elements                    
                    }
                foreach (var line in data)//For each admission
                {
                    mss += String.Join(",", line.Skip(skipDates).ToArray()) + ",";
                }
                if (futurePadding)//If future padding, add the zeros at the other end.
                    for (int i = 0; i < 32 - data.Count; i++)
                    {
                        mss += String.Join(",", data[0].Skip(skipDates).Select(x => 0).ToArray()) + ",";//Zeros for the number of elements                    
                    }
            }

            return mss.TrimEnd(',');
        }

        public bool Mortality { get => !String.IsNullOrEmpty(dateOfDeath); }


        public List<List<string>> GetData(bool OneHotCode, out List<TemporalAbstraction> mss, out List<TemporalAbstractionInterval> mssIntervalList,
        Dictionary<string, int> diagnosisDic, Dictionary<string, int> procedureDic, bool future)
        {
            {
                {
                    {
                        List<List<string>> lists = new List<List<string>>();
                        mss = new List<TemporalAbstraction>();
                        mssIntervalList = new List<TemporalAbstractionInterval>();

                        //Give a line
                        foreach (var admission in admissions)
                        {
                            List<TemporalAbstractionInterval.Variable> fieldVariables = new List<TemporalAbstractionInterval.Variable>();
                            List<string> list = new List<string>();
                            double admissionTime = 0;
                            double dischargeTime = 0;
                            if (!future)
                            {
                                admissionTime = (-(DateTime.MaxValue - admission.AdmissionDate).TotalSeconds);

                                dischargeTime = (-(DateTime.MaxValue - admission.DischargeDate).TotalSeconds);
                            }
                            else//If future
                            {
                                admissionTime = (admission.AdmissionDate - DateTime.MinValue).TotalSeconds;

                                dischargeTime = (admission.DischargeDate - DateTime.MinValue).TotalSeconds;
                            }
                            list.Add(admissionTime.ToString());
                            list.Add(dischargeTime.ToString());
                            //list.Add("");
                            if (!OneHotCode)
                                foreach (var diagnosis in admission.diagnoses)
                                {
                                    list.Add(diagnosisDic[diagnosis.ICD9].ToString());//Code
                                }
                            else
                            //If slow, extract the index keys once, and output zero until reaching each index point
                            {
                                var indices = diagnosisDic.Values.ToArray();
                                var oneHotCode = new int[diagnosisDic.Values.Count];

                                foreach (var diag in admission.diagnoses)//Loop through all the admission list
                                {
                                    oneHotCode[diagnosisDic[diag.ICD9]] = 1;
                                    //list.Add(admission.diagnoses.Exists(x => x.ICD9 == diag.Key) ? "1" : "0");//If that diagnosis is in the admission list, put 1, else put 0                                        
                                }
                                foreach (var index in indices)
                                {
                                    //Should work! If not, something is wrong somewhere!
                                    list.Add(oneHotCode[index].ToString());
                                    mss.Add(new TemporalAbstraction(index, oneHotCode[index], (float)admissionTime, (float)dischargeTime));//If that diagnosis is in the admission list, put 1, else put 0
                                    fieldVariables.Add(new TemporalAbstractionInterval.Variable(index, oneHotCode[index]));
                                }


                                //ORIGINAL WORKING CODE
                                //foreach (var diag in diagnosisDic)//Loop through all the admission list
                                //{
                                //list.Add(admission.diagnoses.Exists(x => x.ICD9 == diag.Key) ? "1" : "0");//If that diagnosis is in the admission list, put 1, else put 0
                                //mss.Add(new TemporalAbstraction(diag.Value, admission.diagnoses.Exists(x => x.ICD9 == diag.Key) ? 1 : 0, (float)admissionTime, (float)dischargeTime));//If that diagnosis is in the admission list, put 1, else put 0
                                //}
                            }

                            //list.Add("");
                            if (!OneHotCode)
                                foreach (var procedure in admission.procedures)
                                {
                                    list.Add(procedureDic[procedure.ICD9].ToString());//Code

                                }
                            else
                            //If slow, extract the index keys once, and output zero until reaching each index point
                            {
                                var indices = procedureDic.Values.ToArray();
                                var oneHotCode = new int[procedureDic.Values.Count];

                                foreach (var proc in admission.procedures)//Loop through all the admission list
                                {
                                    oneHotCode[procedureDic[proc.ICD9]] = 1;
                                    //list.Add(admission.diagnoses.Exists(x => x.ICD9 == diag.Key) ? "1" : "0");//If that diagnosis is in the admission list, put 1, else put 0                                        

                                }
                                foreach (var index in indices)
                                {
                                    //Should work! If not, something is wrong somewhere!
                                    list.Add(oneHotCode[index].ToString());
                                    mss.Add(new TemporalAbstraction(diagnosisDic.Count + index, oneHotCode[index], (float)admissionTime, (float)dischargeTime));//If that diagnosis is in the admission list, put 1, else put 0
                                    fieldVariables.Add(new TemporalAbstractionInterval.Variable(diagnosisDic.Count + index, oneHotCode[index]));
                                }
                            }
                            //ORIGINAL WORKING CODE
                            /*
                                foreach (var proc in procedureDic)//Loop through all the admission list
                                {
                                    list.Add(admission.procedures.Exists(x => x.ICD9 == proc.Key) ? "1" : "0");//If that diagnosis is in the admission list, put 1, else put 0
                                    mss.Add(new TemporalAbstraction(diagnosisDic.Count+  proc.Value, admission.procedures.Exists(x => x.ICD9 == proc.Key) ? 1 : 0, (float)admissionTime, (float)dischargeTime));//If that diagnosis is in the admission list, put 1, else put 0

                                }
                                */


                            lists.Add(list);
                            mssIntervalList.Add(new TemporalAbstractionInterval(fieldVariables, (float)admissionTime, (float)dischargeTime));
                        }

                        return lists;
                    }

                }
            }
        }

        internal bool ValidData()
        {
            if (String.IsNullOrEmpty(deathTime) && String.IsNullOrEmpty(dateOfDeath))
                //If both death times are empty, we should NOT HAVE an empty discharge time, i.e, LastTime being equal to a one of the discharge times!
                if (LastDischargeOrDeathTime() == DateTime.MinValue)
                    //That means both a discharge time and death value was empty. Probably the patient has died, with no death time. Should be checked. OR, missing value
                    return false;
            //Also, last date time should be greater than all admission dates
            foreach (var admi in admissions)
            {
                if (admi.AdmissionDate.CompareTo(LastDischargeOrDeathTime()) > 0)
                    return false;
            }

            return true;


        }



        internal bool OmitAndPredict(int admissionsToOmmit, int predictionLength, string predictionUnit, out Patient patientFuturePrediction)
        {
            //Remove the last n admissions
            //If patient has <admissionsToOmmit admissions, patient is invalid


            patientFuturePrediction = Form1.DeepClone(this);

            patientFuturePrediction.admissions = (from a in admissions where a.DischargeDate == admissions.Max(x => x.DischargeDate) select a).ToList();//Just the final admission


            if (admissions.Count <= admissionsToOmmit)
                return false;

            if (!admissions.Remove((from a in admissions where a.DischargeDate == admissions.Max(x => x.DischargeDate) select a).FirstOrDefault()))
                throw new Exception("Could not remove admissino");

            var secondlastAdmissionOfPatient = (from a in admissions where a.DischargeDate == admissions.Max(x => x.DischargeDate) select a).FirstOrDefault();
            var justRemoveAdmissionForFutureUse = patientFuturePrediction.admissions.FirstOrDefault();

            justRemoveAdmissionForFutureUse.AdmissionDate = DateTime.MinValue + (justRemoveAdmissionForFutureUse.AdmissionDate - secondlastAdmissionOfPatient.AdmissionDate);//0+ the difference of the previous datetime
            try
            {

                justRemoveAdmissionForFutureUse.DischargeDate = DateTime.MinValue + (justRemoveAdmissionForFutureUse.DischargeDate - secondlastAdmissionOfPatient.DischargeDate);//0+ the difference of the previous datetime
            }
            catch (Exception)
            {

                throw;
            }




            //Should move it to future, right? 
            //Yes, just the negative of the other one.

            //Because everything is normalized to DateTime.Max
            //DateOfDeath = DateTime.MaxValue;

            //Normalize admission times (should change the deathtime to future)            

            return true;
        }

        /// <summary>
        /// Creates an array of size 'periods' in future, representing the periods the patient will not be alive in future.
        /// 
        /// </summary>
        /// <param name="periods"></param>
        /// <returns></returns>
        internal List<int> MortalityArray(int periods, bool singleRegression = false)//, TimeSpan timeUnit, float delta)
        {
            if (!Mortality)
                return (from n in Enumerable.Range(0, periods) select 0).ToList();//All zeros for future!

            NormalizeAdmissionTimes(TimeSpan.Zero);//if not done already
            List<int> list = new List<int>();
            var ps = LifeModel.LifeModel.CreatePeriods(periods, future: true);
            var lastAdmissionDischargeTime = admissions.Max(x => x.DischargeDate);
            var deathTimeInFuture = DateOfDeath.Value - lastAdmissionDischargeTime;
            //if (deathTimeInFuture.Ticks < 1)
            //    throw new Exception("Logic exception. LastDischargeOrDeathTime won't give DeathTime");
            var futureDeathMonth = deathTimeInFuture.TotalDays / 30.0;
            for (int i = 0; i < periods; i++)
            {

                //Create the periods first (copy paste)
                //Make sure the units/timespans are correct//Anything months!
                //Then:
                //if deathtime is within the reach of period [i], 

                //ASSUMPTION: LAST ADMISSION IS REMOVED

                if (futureDeathMonth < ps[i].end)//If the person died before the end of that period
                {
                    if (singleRegression)
                    {
                        list.Add(i);//Return the first period the person died
                        return list;
                    }
                    else
                        list.Add(1);
                    //for (int j = 0; j <= periods - list.Count; j++)//Fillout the list with 1s
                    //{
                    //    list.Add(1);
                    //}
                    //break;
                }
                else
                    list.Add(0);

            }

            return list;
        }
    }

    [Serializable]
    public class Admission
    {
        //public string admissionID;
        public string AdmissionID { get; set; }
        private string admissionDate;
        private string dischargeDate;
        public List<Procedure> procedures;
        //public HashSet<Procedure> prodecuresSet;
        //public HashSet<Diagnosis> diagnosisSet;
        public List<Diagnosis> diagnoses;

        public Admission()
        {
            diagnoses = new List<Diagnosis>();
            procedures = new List<Procedure>();
        }

        public Admission(string admissionID, string admissionDate, string dischargeDate)
        {
            this.admissionDate = admissionDate;
            this.AdmissionID = admissionID;
            this.dischargeDate = dischargeDate;
            diagnoses = new List<Diagnosis>();
            procedures = new List<Procedure>();
        }

        public DateTime AdmissionDate { get => DateTime.Parse(admissionDate); set => admissionDate = value.ToString(); }

        /// <summary>
        /// If dischargeDate is Null or Empty, returns DateTime.MinValue
        /// </summary>
        public DateTime DischargeDate
        {
            get
            {
                if (String.IsNullOrEmpty(dischargeDate))
                    return DateTime.MinValue;
                return DateTime.Parse(dischargeDate);
            }
            set => dischargeDate = value.ToString();
        }

    }

    [Serializable]
    public class Diagnosis
    {
        //public string ICD9;
        public string ICD9 { get; set; }
        public string Sequence { get; set; }
        //public string sequence;
        public Diagnosis(string ICD9)
        {
            this.ICD9 = ICD9;
        }
        public Diagnosis(string ICD9, string sequence)
        {
            this.ICD9 = ICD9;
            this.Sequence = sequence;
        }
    }
    [Serializable]
    public class Procedure
    {
        public string ICD9 { get; set; }
        public string Sequence { get; set; }
        //public string sequence;
        public Procedure(string ICD9)
        {
            this.ICD9 = ICD9;
        }
        public Procedure(string ICD9, string sequence)
        {
            this.ICD9 = ICD9;
            this.Sequence = sequence;
        }
    }

}

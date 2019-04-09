using MIMICIIIReader.LifeModel;
using MIMICIIIReader.Models;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MIMICIIIReader
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        static int noOfPatients = 0;
        static Dictionary<string, int> dictionary = new Dictionary<string, int>();
        static bool cancel = false;
        //static List<Patient> patients = new List<Patient>();
        static Dictionary<string, Patient> patientsDic = new Dictionary<string, Patient>();
        static Dictionary<string, int> diagnosisDic = new Dictionary<string, int>();
        static Dictionary<string, int> procedureDic = new Dictionary<string, int>();
        private async void btn_Read_Click(object sender, EventArgs e)
        {
            cancel = false;
            timer1.Enabled = true;
            //TextReader tr = new StreamReader("MIMICIII.csv");//Sample test
            //string line = tr.ReadLine();            
            await Task.Factory.StartNew(Map, TaskCreationOptions.LongRunning);
            foreach (var p in patientsDic.Select(x => x.Value))
            {
                if (dictionary.Keys.Contains(p.admissions.Count.ToString()))
                    dictionary[p.admissions.Count.ToString()] += 1;
                else
                    dictionary.Add(p.admissions.Count.ToString(), 1);
                //dictionary[p.admissions.Count.ToString()] = 1;
            }
            richTextBox1.Text += string.Join(Environment.NewLine, dictionary.Select(x => x.ToString()).ToArray());
        }

        /// <summary>
        /// Reads the patients data from MIMICALL.csv and populates the PatientDic, DiagnosisDic, and ProcedureDic
        /// Each patient contains SubjectID, AdmissionID, AdmissionTime, DischargeTime, DeathTime (And DateOfDeath for the day the death occured), Diagnosis ICD9 Code, ProcedureICD9code, and their sequences
        /// </summary>
        /// <returns></returns>
        private static Task<int> Map()
        {
            TextReader tr = new StreamReader("MIMICALL.csv");//All DATA from MIMIC III            
            foreach (var record in tr.ReadToEnd().Split('\n'))
            {
                if (record.Length < 3) break;
                if (cancel)
                    break;
                string[] columns = record.Split(',');
                var subjectID = columns[0];
                var admissionID = columns[1];
                var admitTime = columns[2];
                var dischargeTime = columns[3];
                var deathTime = columns[4];
                var dateOfDeath = columns[5];//Midnight (0:00 AM) of the deathTime
                var diagnosis = columns[7];
                var diagnosisSequence = columns[8];
                var procedure = columns[9];
                var procedureSequence = columns[10];

                if (!diagnosisDic.ContainsKey(diagnosis))
                    diagnosisDic.Add(diagnosis, diagnosisDic.Count);

                if (!procedureDic.ContainsKey(procedure))
                    procedureDic.Add(procedure, procedureDic.Count);



                //If a patient is there, retrieve it. If not, create it
                //var p = patients.Find(x => x.subjectID == subjectID);
                Patient p = null;
                //if (p == null)
                if (!patientsDic.ContainsKey(subjectID))
                {
                    //Create patient
                    p = new Patient() { subjectID = subjectID, dateOfDeath = dateOfDeath, deathTime = deathTime, admissions = new List<Admission>() };

                    //Create admission
                    //Admission ad = new Admission() { admissionID = admissionID, dischargeDate = dischargeTime, admissionDate = admitTime, /*diagnosisSet = new HashSet<Diagnosis>(), prodecuresSet = new HashSet<Procedure>(),*/ diagnoses = new List<Diagnosis>(), procedures = new List<Procedure>() };
                    Admission ad = new Admission(admissionID, admitTime, dischargeTime);


                    //Add diagnosis
                    Diagnosis diag = new Diagnosis(ICD9: diagnosis, sequence: diagnosisSequence);

                    //Add prodecure
                    Procedure proc = new Procedure(ICD9: procedure, sequence: procedureSequence);



                    ad.diagnoses.Add(diag);
                    ad.procedures.Add(proc);


                    //ad.diagnosisSet.Add(diag);
                    //ad.prodecuresSet.Add(proc);

                    p.admissions.Add(ad);

                    //patients.Add(p);
                    patientsDic.Add(p.subjectID, p);
                }
                else//Patient already exists. Check if we have the admission or not
                {
                    p = patientsDic[subjectID];
                    var ad = p.admissions.Find(x => x.AdmissionID == admissionID);
                    bool admissionExists = true;
                    if (ad == null)
                    {
                        admissionExists = false;
                        //Create new admission
                        //ad = new Admission() { admissionID = admissionID, dischargeDate = dischargeTime, admissionDate = admitTime, /*diagnosisSet = new HashSet<Diagnosis>(), prodecuresSet = new HashSet<Procedure>(),*/ diagnoses = new List<Diagnosis>(), procedures = new List<Procedure>() };
                        ad = new Admission(admissionID, admitTime, dischargeTime);
                    }


                    //Add diagnosis
                    Diagnosis diag = new Diagnosis(ICD9: diagnosis, sequence: diagnosisSequence);

                    //Add prodecure
                    Procedure proc = new Procedure(ICD9: procedure, sequence: procedureSequence);

                    if (ad.diagnoses.Find(x => x.ICD9.Equals(diag.ICD9)) == null)
                        ad.diagnoses.Add(diag);

                    if (ad.procedures.Find(x => x.ICD9.Equals(proc.ICD9)) == null)
                        ad.procedures.Add(proc);

                    //Add data to admission                    
                    //ad.diagnosisSet.Add(diag);
                    //ad.prodecuresSet.Add(proc);

                    if (!admissionExists)
                        p.admissions.Add(ad);
                    else//Update admissions
                        p.admissions[p.admissions.FindIndex(a => a.AdmissionID.Equals(ad.AdmissionID))] = ad;

                    patientsDic[subjectID] = p;
                }
                noOfPatients = patientsDic.Count;

            }

            return Task.FromResult(noOfPatients);
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            this.Text = $"Patients: {patientsDic.Count}, Procedures: {procedureDic.Count}, Diagnosis: {diagnosisDic.Count}";
        }

        private void btn_Cancel_Click(object sender, EventArgs e)
        {
            cancel = true;
        }

        private void btn_GetPatient_Click(object sender, EventArgs e)
        {
            richTextBox1.ResetText();
            //richTextBox1.Text = String.Join(",", patientsDic.First().Value.GetData(false).Select(x => String.Join(",", x.ToArray())).ToArray());

            richTextBox1.Text += Environment.NewLine + "OUTPUT:" + Environment.NewLine;

            // richTextBox1.Text += patientsDic.First().Value.ConvertToMSS(false);
            richTextBox1.Text += Environment.NewLine + "ONEHOTCODE:" + Environment.NewLine;
            // richTextBox1.Text += patientsDic.First().Value.ConvertToMSS(true);

            // richTextBox1.Text = patientsDic.First().Value.GetData().Select(x => x.ToArray().ToString()).ToArray().ToString();
        }

        private void btn_NormalizeTime_Click(object sender, EventArgs e)
        {
            Patient firstPatient = patientsDic.First().Value;
            richTextBox1.Text += Environment.NewLine;
            richTextBox1.Text += "Admission Dates: " + String.Join(", ", firstPatient.admissions.Select(x => x.AdmissionDate).ToArray()) + Environment.NewLine;
            richTextBox1.Text += "Discharge Dates: " + String.Join(", ", firstPatient.admissions.Select(x => x.DischargeDate).ToArray()) + Environment.NewLine;
            richTextBox1.Text += $"Last admission date:{patientsDic.First().Value.LastDischargeOrDeathTime()}, previous admission date{patientsDic.First().Value.LastDischargeOrDeathTime()}";
            patientsDic.First().Value.NormalizeAdmissionTimes(TimeSpan.Zero);

            //See what changed
            firstPatient = patientsDic.First().Value;
            richTextBox1.Text += Environment.NewLine;
            richTextBox1.Text += "Admission Dates: " + String.Join(", ", firstPatient.admissions.Select(x => x.AdmissionDate).ToArray()) + Environment.NewLine;
            richTextBox1.Text += "Discharge Dates: " + String.Join(", ", firstPatient.admissions.Select(x => x.DischargeDate).ToArray()) + Environment.NewLine;
            richTextBox1.Text += $"Last admission date:{patientsDic.First().Value.LastDischargeOrDeathTime()}, previous admission date{patientsDic.First().Value.LastDischargeOrDeathTime()}";
        }

        public static Options GetOptions { get; set; } = new Options();

        public enum FutureMortalityMethod
        {
            SingleRegression,
            BinaryPeriods,
            Seq2SeqDiagnosisLM
        }

        public class Options
        {
            /// <summary>
            /// Whether or not ommit the last admission to predict future mortality
            /// </summary>
            public bool OmitLastOmission { get; set; } = true;
            public bool FixedWidnowsOutput { get; set; } = false;

            /// <summary>
            /// Whether or not write the "Reg" data. This data is the list of admissions per line. For two admissions:
            /// mortality (1,0),admission1_admissionTime,admission1_dischargetime,diagnosis(OneHotCode), procedures (OneHotCode),admission2_admissionTime,admission2_dischargetime,diagnosis(OneHotCode), procedures (OneHotCode),
            /// </summary>
            public bool WriteRegularOutput { get; set; } = true;

            /// <summary>
            /// Whether or not write 0 or 1 prediction bit before Regular data
            /// </summary>
            public bool WriteMortalityBitAtTheStartOfRegularData { get; set; } = false;

            /// <summary>
            /// Whether or not add padding to regular output to make it all length 32
            /// </summary>
            public bool PadRegularOutputTo32 { get; set; } = true;

            /// <summary>
            /// Whether or not write the input file for LM or Fixed ITS
            /// </summary>
            public bool WriteLMorFixedData { get; set; } = true;

            /// <summary>
            /// 
            /// </summary>
            public FutureMortalityMethod FutureMortalityWriteMethod { get; set; } = FutureMortalityMethod.SingleRegression;

            public int PeriodsForMortalityForecasting { get; set; } = 4;

            public int MaxSamples { get; set; } = -1;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            propertyGrid1.SelectedObject = GetOptions;
            //GetOptions.MaxSamples = 50;
            GetOptions.FutureMortalityWriteMethod = FutureMortalityMethod.Seq2SeqDiagnosisLM;
            btn_Read_Click(null, null);
            //btn_ReadFallData_Click_1(null, null);
        }

        private async void btn_WriteOutput_Click(object sender, EventArgs e)
        {
            timer1.Enabled = false;
            var progressIndicator = new Progress<(int patients, int invalid)>(((int patients, int invalid) t) => { this.Text = $"Patients written to disk={t.patients}, invalid={t.invalid}"; });
            await Task.Factory.StartNew(new Action(() => { WriteDataToFile(progressIndicator, GetOptions.WriteRegularOutput, GetOptions.WriteLMorFixedData, GetOptions.FixedWidnowsOutput, GetOptions.OmitLastOmission, GetOptions.FutureMortalityWriteMethod, GetOptions.PeriodsForMortalityForecasting); }), TaskCreationOptions.None);
            //Task.Factory.StartNew<int>(WriteDataToFile, progressIndicator, TaskCreationOptions.LongRunning);
        }

        /// <summary>
        /// Write the data used for learning, both input and output
        /// </summary>
        /// <param name="progress">For progress repo`rt (no of patients completed)</param>
        /// <param name="writeRegular">Whether or not write the "Reg" data. This data is the list of admissions per line. For two admissions:
        /// mortality (1,0),admission1_admissionTime,admission1_dischargetime,diagnosis(OneHotCode), procedures (OneHotCode),admission2_admissionTime,admission2_dischargetime,diagnosis(OneHotCode), procedures (OneHotCode),
        ///</param>        
        /// <param name="writeLMorFixedData">Whether or not write the input file for LM or Fixed ITS</param>
        /// <param name="fixedWindows">Whether to use LM mapping or Fixed period mappings</param>
        /// <param name="omitLastAdmission">Whether or not ommit the last admission to predict future mortality</param>
        private static void WriteDataToFile(IProgress<(int, int)> progress, bool writeRegular = true, bool writeLMorFixedData = true, bool fixedWindows = false/*Fixed Sized vs LM*/, bool omitLastAdmission = true, FutureMortalityMethod futureMortalityMethod = FutureMortalityMethod.BinaryPeriods,
            int futureMortalityPeriods = 4)
        {
            //TextWriter ts_LM_Input = File.CreateText($"MIMICIII_Diag_Proc_LM_Input{DateTime.Now.ToString()}.csv".Replace(":", "-").Replace("/", "-"));            
            //TextWriter ts_LM_Input = new StreamWriter($"MIMICIII_Diag_Proc_LM_Input{DateTime.Now.ToString()}.csv".Replace(":", "-").Replace("/", "-"));//, false, Encoding.Default, 65536000);
            //Stream inputStream=File.Create()
            var fileLabel = (fixedWindows ? "Fixed" : "LM");

            string fileName = $"MIMICIII_Diag_Proc_{fileLabel}{(omitLastAdmission ? ("_Forecast_" + futureMortalityMethod.ToString() + futureMortalityPeriods.ToString()) : "")}_Input{DateTime.Now.ToString()}.csv.gz".Replace(":", "-").Replace("/", "-");
            string outputFilename = fileName.Replace("Input", "Output");


            Stream s = new FileStream(fileName, FileMode.Create);
            var gZipStream = new GZipStream(s, CompressionMode.Compress);//, false, Encoding.Default, 65536000);
            TextWriter ts_LM_Input = new StreamWriter(gZipStream);

            TextWriter ts_LM_Output = null;
            Stream sOut = null;
            GZipStream gZipStreamOut = null;//= new GZipStream(sOut, CompressionMode.Compress);//, false, Encoding.Default, 65536000);
            if (futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM)
            {
                sOut = new FileStream(outputFilename, FileMode.Create);
                gZipStreamOut = new GZipStream(sOut, CompressionMode.Compress);//, false, Encoding.Default, 65536000);
                ts_LM_Output = new StreamWriter(gZipStreamOut);
            }
            else
                ts_LM_Output = File.CreateText(outputFilename.Replace(".csv.gz", ".csv"));


            TextWriter ts_Reg_Input = null;
            Stream s_reg = null;
            GZipStream gZipStream_Reg = null;
            if (writeRegular)
            {
                s_reg = new FileStream(fileName.Replace(fileLabel, "Reg"), FileMode.Create);
                gZipStream_Reg = new GZipStream(s_reg, CompressionMode.Compress);//, false, Encoding.Default, 65536000);
                ts_Reg_Input = new StreamWriter(gZipStream_Reg);// File.CreateText($"MIMICIII_Diag_Proc_{DateTime.Now.ToString()}.csv".Replace(":", "-").Replace("/", "-"));
            }

            int invalidPatients = 0;
            int counter = 0;
            List<List<TemporalAbstraction>> mss = new List<List<TemporalAbstraction>>();
            foreach (var patient in patientsDic)
            {
                Patient p = patient.Value;

                //p.NormalizeAdmissionTimes(TimeSpan.Zero);
                List<TemporalAbstraction> mssP = new List<TemporalAbstraction>();
                List<TemporalAbstractionInterval> mssIntervalP = new List<TemporalAbstractionInterval>();

                Patient pFuturePrediction = DeepClone(patient.Value);
                List<TemporalAbstraction> mssPFuture = new List<TemporalAbstraction>();
                List<TemporalAbstractionInterval> mssIntervalPFuture = new List<TemporalAbstractionInterval>();

                if (p.ValidData() && (omitLastAdmission ? p.OmitAndPredict(1, 4, "Month", out pFuturePrediction) : true))
                {
                    //if (writeVariableNonLM)
                    //List<string> result = p.ConvertToOneHotArray(true, diagnosisDic, procedureDic);
                    var dataString = p.ConvertToMSS(true, out mssP, out mssIntervalP, diagnosisDic, procedureDic,
                        GetOptions.WriteMortalityBitAtTheStartOfRegularData, GetOptions.PadRegularOutputTo32,
                        futurePadding: false);

                    string dataStringFuture = null;
                    if (futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM)

                        dataStringFuture = pFuturePrediction.ConvertToMSS(true, out mssPFuture, out mssIntervalPFuture, diagnosisDic, procedureDic,
                        GetOptions.WriteMortalityBitAtTheStartOfRegularData, GetOptions.PadRegularOutputTo32,
                        futurePadding: futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM);

                    //Shift the mssIntervalPFuture with the difference between the two....but don't have data anymore, do I?

                    if (writeRegular)
                    {

                        ts_Reg_Input.WriteLine(dataString);
                    }

                    if (writeLMorFixedData)
                    {
                        LifeModel.LifeModel lm = new LifeModel.LifeModel(8371, 2, 5, false, true);
                        LifeModel.LifeModel lmFuture = new LifeModel.LifeModel(8371, 2, 5, true, true);

                        //List<TemporalAbstractionInterval> temporalAbstractionInterval = TemporalAbstractionInterval.FromTemporalAbstractions(mssP);
                        //var patientLM = lm.GetITS(mssP, false);
                        //var patientLM = lm.GetITS(temporalAbstractionInterval, false);
                        double[,,] patientLM = lm.GetITS(mssIntervalP, fixedWindows);
                        double[,,] patientLMFuture = null;

                        if (futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM)
                            patientLMFuture = lmFuture.GetITS(mssIntervalPFuture, fixedWindows);
                        //TODO: Write patient lifeModel
                        //String line = "";
                        //bool sparse = false;
                        //if (!sparse)
                        {
                            StringBuilder sb_LM_Input = new StringBuilder();
                            StringBuilder sb_LM_OutputFuture = new StringBuilder();

                            //var len = patientLM.Length;
                            foreach (var i in patientLM)
                            {
                                if (i == (int)i)//It was 0, just output 0. Fixes 0.0000000001 problem maybe?
                                    // ts_LM_Input.Write((int)i + ",");
                                    sb_LM_Input.Append((int)i + ",");


                                else//Write non-zero
                                    //ts_LM_Input.Write(i + ",");
                                    sb_LM_Input.Append(i + ",");

                                //ts_LM_Input.Write(",");
                                //    //(int)patientLM[k, f, v] + ",");
                            }
                            if (futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM)
                                foreach (var i in patientLMFuture)
                                {
                                    if (i == (int)i)//It was 0, just output 0. Fixes 0.0000000001 problem maybe?
                                                    // ts_LM_Input.Write((int)i + ",");
                                        sb_LM_OutputFuture.Append((int)i + ",");


                                    else//Write non-zero
                                        //ts_LM_Input.Write(i + ",");
                                        //sb_LM_OutputFuture.Append(i + ",");
                                        sb_LM_OutputFuture.Append(1 + ",");//For diagnosis prediction only

                                    //ts_LM_Input.Write(",");
                                    //    //(int)patientLM[k, f, v] + ",");
                                }
                            //ts_LM_Input.WriteLine();
                            //ts_LM_Input.WriteLine(String.Join(",", patientLM.GetEnumerator().));
                            ts_LM_Input.Write(sb_LM_Input.ToString().TrimEnd(','));
                            if (futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM)
                                ts_LM_Output.Write(sb_LM_OutputFuture.ToString().TrimEnd(','));
                        }

                        ////if (sparse)
                        ////{
                        ////    ts_LM_Input.WriteLine(counter);
                        ////    for (int k = 0; k < patientLM.GetLength(0); k++)
                        ////    {
                        ////        for (int f = 0; f < patientLM.GetLength(1); f++)
                        ////        {
                        ////            for (int v = 0; v < patientLM.GetLength(2); v++)
                        ////            {
                        ////                double value = patientLM[k, f, v];
                        ////                if (value != 0.0d)//Write Sparse
                        ////                    ts_LM_Input.WriteLine($"{k},{f},{v},{value}");

                        ////                //            //if (patientLM[k, f, v].Equals((int)patientLM[k, f, v]))//If it is zero or 1, save some string space and write int instead of float (0 vs 0.0)
                        ////                //            //  ts_LM_Input.Write((int)patientLM[k, f, v] + ",");
                        ////                //            //line += (int)patientLM[k, f, v] + ",";
                        ////                //            //else
                        ////                //            //line += patientLM[k, f, v] + ",";
                        ////            }
                        ////        }
                        ////    }
                        ////}
                        //ts_LM_Input.WriteLine(line.TrimEnd(','));//Remove the last ,
                        ts_LM_Input.WriteLine();//End line
                        if (futureMortalityMethod == FutureMortalityMethod.Seq2SeqDiagnosisLM)
                            ts_LM_Output.WriteLine();
                        if (!omitLastAdmission)
                            ts_LM_Output.WriteLine(patient.Value.Mortality ? "1" : "0");//Remove the last
                        else
                        {
                            if (futureMortalityMethod == FutureMortalityMethod.BinaryPeriods)
                                ts_LM_Output.WriteLine(string.Join(",", patient.Value.MortalityArray(periods: futureMortalityPeriods, singleRegression: false/*, timeUnit: new TimeSpan(30,0,0,0), delta: 1.0f*/).ToArray()));//Remove the last
                            else if (futureMortalityMethod == FutureMortalityMethod.SingleRegression)
                            {
                                if (!patient.Value.Mortality)
                                    ts_LM_Output.WriteLine(-1);
                                else
                                    ts_LM_Output.WriteLine(patient.Value.MortalityArray(periods: futureMortalityPeriods, singleRegression: true/*, timeUnit: new TimeSpan(30,0,0,0), delta: 1.0f*/).Last());//Remove the last
                            }
                            else
                                ;//Do nothing, I guess
                        }
                    }
                }
                else
                    invalidPatients++;
                counter++;
                progress.Report((counter, invalidPatients));
                //if (counter - invalidPatients >= 500)
                //    break;
                if (GetOptions.MaxSamples > 0)
                    if (counter > GetOptions.MaxSamples)
                        break;
            }
            if (writeRegular)
                ts_Reg_Input.Close();
            ts_LM_Input.Close();
            ts_LM_Output.Close();
            MessageBox.Show("Done! Invalid Data " + invalidPatients);
            //return Task.FromResult<int>(counter);
        }

        public static T DeepClone<T>(T obj)
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, obj);
                ms.Position = 0;

                return (T)formatter.Deserialize(ms);
            }
        }
        private void btn_LifeModel_Click(object sender, EventArgs e)
        {

            string LM_Input = "";
            LM_Input = PatientLMSample();

        }

        private static string PatientLMSample()
        {
            string LM_Input = "";
            //TestLifeModel(2, 2);
            //Get MSS
            //int F = 8372;
            //LifeModel.LifeModel lm = new LifeModel.LifeModel(F, 2, 5);
            //var MSS=GetDa
            //var ITS = lm.GetITS(MSS);

            //Admission ad1=new Admission() { }




            //Admission ad1 = new Admission()
            //{
            //    AdmissionID = "1",
            //    AdmissionDate = DateTime.Now.Subtract(new TimeSpan(10, 0, 0, 0)),
            //    DischargeDate = DateTime.Now.Subtract(new TimeSpan(8, 0, 0, 0)),
            //    diagnoses = new List<Diagnosis>() { },
            //    procedures = new List<Procedure>() { }
            //};


            //Patient 9370	Admission 160150
            Admission ad2 = new Admission()
            {
                AdmissionID = "1",
                AdmissionDate = DateTime.Now.Subtract(new TimeSpan(3, 0, 0, 0)),
                DischargeDate = DateTime.Now.Subtract(new TimeSpan(1, 0, 0, 0)),
                diagnoses = new List<Diagnosis> { new Diagnosis("9974"), new Diagnosis("99859"), new Diagnosis("5109") },
                procedures = new List<Procedure> { new Procedure("3404"), new Procedure("9915") },
            };


            Patient p = new Patient()
            {
                subjectID = "1",
                deathTime = DateTime.Now.Subtract(new TimeSpan(1, 0, 0, 0)).ToString(),
                dateOfDeath = DateTime.Now.Subtract(new TimeSpan(1, 0, 0, 0)).ToString(),
                admissions = new List<Admission>() { ad2 }
            };


            var fixedWindows = false;
            StringBuilder LM_Input_SB = new StringBuilder();

            //Patient p = patient.Value;
            List<TemporalAbstraction> mssP = new List<TemporalAbstraction>();
            List<TemporalAbstractionInterval> mssIntervalP = new List<TemporalAbstractionInterval>();
            if (p.ValidData())
            {
                var dataString = p.ConvertToMSS(true, out mssP, out mssIntervalP, diagnosisDic, procedureDic);
                //if (writePatientData)
                //    ts.WriteLine(dataString);
                //if (writeLM)
                {
                    LifeModel.LifeModel lm = new LifeModel.LifeModel(8371, 2, 5, false, true);
                    //List<TemporalAbstractionInterval> temporalAbstractionInterval = TemporalAbstractionInterval.FromTemporalAbstractions(mssP);
                    //var patientLM = lm.GetITS(mssP, false);
                    //var patientLM = lm.GetITS(temporalAbstractionInterval, false);
                    var patientLM = lm.GetITS(mssIntervalP, fixedWindows);
                    //TODO: Write patient lifeModel
                    String line = "";
                    bool sparse = false;
                    if (!sparse)
                    {
                        var len = patientLM.Length;
                        foreach (var i in patientLM)
                        {
                            if (i == (int)i)
                                LM_Input_SB.Append((int)i + ",");


                            else
                                LM_Input_SB.Append(i + ",");
                            //ts_LM_Input.Write(",");
                            //    //(int)patientLM[k, f, v] + ",");
                        }
                        //ts_LM_Input.WriteLine();
                        //ts_LM_Input.WriteLine(String.Join(",", patientLM.GetEnumerator().));
                    }
                    if (sparse)
                    {
                        //ts_LM_Input.WriteLine(counter);
                        for (int k = 0; k < patientLM.GetLength(0); k++)
                        {
                            for (int f = 0; f < patientLM.GetLength(1); f++)
                            {
                                for (int v = 0; v < patientLM.GetLength(2); v++)
                                {
                                    double value = patientLM[k, f, v];
                                    if (value != 0.0d)//Write Sparse
                                        LM_Input += ($"{k},{f},{v},{value}") + Environment.NewLine;

                                    //            //if (patientLM[k, f, v].Equals((int)patientLM[k, f, v]))//If it is zero or 1, save some string space and write int instead of float (0 vs 0.0)
                                    //            //  ts_LM_Input.Write((int)patientLM[k, f, v] + ",");
                                    //            //line += (int)patientLM[k, f, v] + ",";
                                    //            //else
                                    //            //line += patientLM[k, f, v] + ",";
                                }
                            }
                        }
                    }
                    //ts_LM_Input.WriteLine(line.TrimEnd(','));//Remove the last ,
                    //ts_LM_Input.WriteLine();//End line
                    //ts_LM_Output.WriteLine(p.Mortality ? "1" : "0");//Remove the last
                }
            }
            LM_Input = LM_Input_SB.ToString();
            return LM_Input;
        }

        void TestLifeModel(int F, int V, int n = 5, bool future = false)
        {
            /*
             * 
'''--------------------------------------Reading sample from thesis
F, V= (3,5)
# C, G, B= (0, 1, 2)
# VL, L, N, H, VH= (0, 1, 2, 3, 4)
Z=[TemporalAbstraction(1,3,1,5),#(G,H,1,4)
   TemporalAbstraction(0,2,2,14),#(C,N,2,14)
   TemporalAbstraction(2,2,4,20),#(B,N,4,20)
   TemporalAbstraction(1,2,6,9),#(G,N,6,9)
   TemporalAbstraction(1,3,10,13),#(G,H,10,13)
   TemporalAbstraction(0,3,15,24),#(C,H,15,24)
   TemporalAbstraction(1,4,16,23)#(G,VH,16,23)
   ]
'''

             * */
            F = 2;
            V = 2;
            // int C=0, G=1, 
            List<TemporalAbstraction> MSS = new List<TemporalAbstraction>();
            for (int i = 0; i < 50; i++)
                MSS.Add(new TemporalAbstraction(i % 2, 1 - i % 2, i, i + 2));
            //Z =[TemporalAbstraction(i % 2, (1 - i % 2), i, i + 2) for i in range(50)]



            LifeModel.LifeModel lm = new LifeModel.LifeModel(F, V, n, future);
            var ITS = lm.GetITS(MSS);
        }

        //string URL = "http://localhost:51221/api/v0.1/predict_mortality";
        string URL = "https://pharms.azurewebsites.net/api/v0.1/predict_mortality";
        static readonly HttpClient client = new HttpClient();
        private async void btn_TestAPI_Click(object sender, EventArgs e)
        {
            string lm = PatientLMSample();
            //MessageBox.Show(lm.Length.ToString());
            var responseString = await client.GetStringAsync(URL);
            MessageBox.Show(responseString);



            var values = new Dictionary<string, string>
            {
   { "ack", "hello" },
   //{ "data", lm}
};

            var content = new FormUrlEncodedContent(values);
            HttpContent c = new StringContent(lm);

            var response = await client.PostAsync(URL, c);// content);

            var responseStringPost = await response.Content.ReadAsStringAsync();
            MessageBox.Show("Post Response: " + responseStringPost);

















            //////HttpWebRequest request = (HttpWebRequest)WebRequest.Create("https://api.github.com/repos/restsharp/restsharp/releases");
            ////HttpWebRequest request = (HttpWebRequest)WebRequest.Create("https://pharms.azurewebsites.net/api/v0.1/predict_mortality");


            ////request.Method = "GET";
            ////request.UserAgent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 58.0.3029.110 Safari / 537.36";
            ////request.AutomaticDecompression = DecompressionMethods.Deflate | DecompressionMethods.GZip;

            ////HttpWebResponse response = (HttpWebResponse)request.GetResponse();

            ////string content = string.Empty;
            ////using (Stream stream = response.GetResponseStream())
            ////{
            ////    using (StreamReader sr = new StreamReader(stream))
            ////    {
            ////        content = sr.ReadToEnd();
            ////    }
            ////}
            ////MessageBox.Show(content);

            //////var releases = JArray.Parse(content);
        }

        public class Sample
        {
            public static bool WriteClass { get; set; } = true;
            public bool Fallen { get; set; }
            public List<AccelRecord> records;
            void Normalize() { throw new NotImplementedException(); }
            void GetLM() { throw new NotImplementedException(); }

            /// <summary>
            /// Creates a single line per sample, started by 1-bit (fall, non-fall) class, and 4 accel data for each record present.
            /// </summary>
            /// <returns>A line representation of sample</returns>
            public override string ToString()
            {
                return WriteClassIfNeeded() + records.Count + "," + string.Join(",", records.Select(x => $"{x.Data.SV},{x.Data.AX},{x.Data.AY},{x.Data.AZ}"));
            }

            /// <summary>
            /// Creates a single line per sample, started by 1-bit (fall, non-fall) class, and 4 accel data for each record present.
            /// </summary>
            /// <returns>A line representation of sample</returns>
            public string ToString(bool AutoShiftToFuture = false)
            {
                if (!AutoShiftToFuture)
                    return WriteClassIfNeeded() + records.Count + "," + string.Join(",", records.Select(x => $"{x.Data.SV},{x.Data.AX},{x.Data.AY},{x.Data.AZ}"));
                else
                {
                    //Get the total time of current sample
                    TimeSpan maxTimeSpan = records.Max(y => y.TimeStamp);
                    double DeltaT = maxTimeSpan.TotalSeconds;
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < DeltaT; i++)
                    {
                        var newRecords = records.Where(r => (r.TimeStamp.TotalSeconds - i) > 0).ToList();
                        sb.AppendLine(WriteClassIfNeeded() + i + "," + (newRecords.Count) + "," + string.Join(",", newRecords.Select(x => $"{x.Data.SV},{x.Data.AX},{x.Data.AY},{x.Data.AZ}")));
                    }

                    return sb.ToString().Trim('\n');
                }
            }

            /// <summary>
            /// Creates a single line per sample, started by 1-bit (fall, non-fall) class, and ONLY 4 accel data, which is the average of the whole data
            /// </summary>
            /// <returns>A line representation of average of acceleration data for this sample</returns>
            internal string ToAverageString()
            {
                return WriteClassIfNeeded() + $"{records.Average(x => x.Data.SV).ToString()}, {records.Average(x => x.Data.AX).ToString()}, {records.Average(x => x.Data.AY).ToString()}, {records.Average(x => x.Data.AZ).ToString()}";
            }

            /// <summary>
            /// Creates a n=4, 16-elemet LM representation of the sample. Returns class (0, 1) + 16x4 array of double in a single line (total 1+16x4=65 CSV elements)
            /// </summary>
            /// <param name="delta"></param>
            /// <param name="fixedIntervals"></param>
            /// <returns></returns>
            public string ToLifeModel(TimeSpan delta, bool fixedIntervals = false, bool Autoshift = false, double missingRate = 0.0, int randSeed = -1)
            {
                var records = CreateRandomSubset(missingRate, randSeed, this.records.ToList());//Create a local variable for records

                if (!Autoshift)
                {
                    LifeModel.LifeModel lm = new LifeModel.LifeModel(4, 1);
                    var result = lm.GetAverageMappedIntervalSequence(records.Select(x => new LifeModel.Point() { TimeStamp = x.TimeStamp, values = new List<double>() { x.Data.SV, x.Data.AX, x.Data.AY, x.Data.AZ } }).ToList(), new AverageMapper(), fixedIntervals: fixedIntervals, delta: delta);
                    return WriteClassIfNeeded() + string.Join(",", result.Cast<double>());// stringBuilder.ToString();
                }
                else
                {
                    ////////recordsList = ;
                    ////////StringBuilder sb = new StringBuilder();
                    ////////foreach (var newRecords in AutoshiftRecords(records.ToList()))
                    ////////{
                    ////////    LifeModel.LifeModel lm = new LifeModel.LifeModel(4, 1);
                    ////////    var result = lm.GetAverageMappedIntervalSequence(records.Select(x => new LifeModel.Point() { TimeStamp = x.TimeStamp, values = new List<double>() { x.Data.SV, x.Data.AX, x.Data.AY, x.Data.AZ } }).ToList(), new AverageMapper(), fixedIntervals: fixedIntervals, delta: delta);


                    ////////    LifeModel.LifeModel lm = new LifeModel.LifeModel(4, 1);
                    ////////    var result = lm.GetAverageMappedIntervalSequence(
                    ////////        newRecords.Select(x => new LifeModel.Point()
                    ////////        {
                    ////////            TimeStamp = x.TimeStamp,
                    ////////            values =
                    ////////        new List<double>() { x.Data.SV, x.Data.AX, x.Data.AY, x.Data.AZ }
                    ////////        }
                    ////////        ).ToList(),
                    ////////        new AverageMapper(), fixedIntervals: fixedIntervals, delta: delta);

                    ////////    sb.AppendLine(WriteClassIfNeeded() + newRecords. + "," + string.Join(",", result.Cast<double>()));

                    ////////}

                    //Get the total time of current sample
                    TimeSpan maxTimeSpan = records.Max(y => y.TimeStamp);
                    double DeltaT = maxTimeSpan.TotalSeconds;
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < DeltaT; i++)
                    {
                        var newRecords = records.Where(r => (r.TimeStamp.TotalSeconds - i) > 0).ToList();
                        LifeModel.LifeModel lm = new LifeModel.LifeModel(4, 1);
                        var result = lm.GetAverageMappedIntervalSequence(
                            newRecords.Select(x => new LifeModel.Point()
                            {
                                TimeStamp = x.TimeStamp,
                                values =
                            new List<double>() { x.Data.SV, x.Data.AX, x.Data.AY, x.Data.AZ }
                            }
                            ).ToList(),
                            new AverageMapper(), fixedIntervals: fixedIntervals, delta: delta);

                        sb.AppendLine(WriteClassIfNeeded() + i + "," + string.Join(",", result.Cast<double>()));
                    }




                    return sb.ToString();//.Trim('\n');
                }
            }

            private static List<AccelRecord> CreateRandomSubset(double missingRate, int randSeed, List<AccelRecord> records)
            {
                if (missingRate > 0)
                {
                    int noOfRecords = records.Count;
                    int recordsToKeep = (int)Math.Min(records.Count, Math.Max(1, Math.Ceiling(noOfRecords * (1 - missingRate))));//Get the missing rate and make sure it is greater than 1 and less than or equal to noOfRecords
                    Random r;
                    if (randSeed > 0)
                        r = new Random(randSeed);
                    else
                        r = new Random();//Random random ?
                    records = records.OrderBy(x => r.Next()).Take(recordsToKeep).ToList();
                    records.Sort((x, y) => (x.TimeStamp.CompareTo(y.TimeStamp)));
                }
                return records;
            }

            private string WriteClassIfNeeded()
            {
                return (WriteClass ? (Fallen ? "1," : "0,") : "");
            }

        }

        public class AccelRecord
        {
            public TimeSpan TimeStamp { get; set; }

            public (double SV, double AX, double AY, double AZ) Data;
            public int Autoshift { get; set; } = 0;
        }


        private async void btn_ReadFallData_Click_1(object sender, EventArgs e)
        {
            cancel = false;
            timer1.Enabled = true;
            //TextReader tr = new StreamReader("MIMICIII.csv");//Sample test
            //string line = tr.ReadLine();            
            await Task.Factory.StartNew(MapFall, TaskCreationOptions.LongRunning);
            foreach (var p in samples.Select(x => (x.records.Max(y => y.TimeStamp)).TotalSeconds))
            {
                //if (dictionary.Keys.Contains(p.admissions.Count.ToString()))
                //  dictionary[p.admissions.Count.ToString()] += 1;
                //else
                //  dictionary.Add(p.admissions.Count.ToString(), 1);
                //dictionary[p.admissions.Count.ToString()] = 1;
                richTextBox1.Text += p.ToString() + Environment.NewLine;

            }
            TimeSpan maxTimeSpan = samples.Min(x => x.records.Max(y => y.TimeStamp));
            double DeltaT = maxTimeSpan.TotalSeconds;
            richTextBox1.Text += $"Min number of seconds in dataset: {DeltaT}";
            //richTextBox1.Text += string.Join(Environment.NewLine, dictionary.Select(x => x.ToString()).ToArray());
        }

        /// <summary>
        /// List of accelererometer samples. 
        /// </summary>
        static List<Sample> samples = new List<Sample>();

        private static Task<int> MapFall()
        {
            string folder = "/CSV";
            var folders = System.IO.Directory.GetFiles(Environment.CurrentDirectory + "\\" + folder);

            foreach (var file in folders)
            {
                TextReader tr = new StreamReader(file);//All DATA from FALL DATA            
                Sample sample = new Sample() { Fallen = file.Contains("adl") };
                List<AccelRecord> records = new List<AccelRecord>();
                var lines = tr.ReadToEnd().Split('\n');
                foreach (var record in lines)
                {
                    if (record.Length < 3) break;
                    if (cancel)
                        break;
                    string[] columns = record.Split(',');
                    AccelRecord ar = new AccelRecord()
                    {
                        TimeStamp = new TimeSpan(0, 0, 0, 0, int.Parse((columns[0]))),
                        Data = (double.Parse(columns[1]),
                        double.Parse(columns[2]),
                        double.Parse(columns[3]),
                        double.Parse(columns[4]))
                    };
                    records.Add(ar);
                    noOfPatients++;
                }
                sample.records = records;
                samples.Add(sample);

                //return Task.FromResult(noOfPatients);
            }

            //double delta = 0;
            TimeSpan maxTimeSpan = samples.Max(x => x.records.Max(y => y.TimeStamp));
            double DeltaT = maxTimeSpan.TotalSeconds;
            //Finding K and N
            int k = 0;
            int n = -1;
            if (n < 0)//it is not defined
            {
                n = 0;
                k = 1;
                int delta = 1;
                while (DeltaT >= Math.Pow(2, delta * k))
                {
                    n++;
                    k = (int)Math.Pow(2, n);
                }
            }
            else//n was provided
                k = (int)Math.Pow(2, n);

            string dateString = DateTime.Now.ToString("yyyy-MM-dd  HH-mm-ss");
            string folderResults = "Fall Data " + dateString;

            Directory.CreateDirectory(folderResults);

            TextWriter ts = new StreamWriter(folderResults + "\\" + $"FallDataAll_{dateString}.csv");
            TextWriter ts2 = new StreamWriter(folderResults + "\\" + $"FallDataAvg_{dateString}.csv");
            TextWriter ts3 = new StreamWriter(folderResults + "\\" + $"FallDataAvgFixed32_{dateString}.csv");
            TextWriter ts4 = new StreamWriter(folderResults + "\\" + $"FallDataAvgLifeModel32_{dateString}.csv");
            TextWriter ts5 = new StreamWriter(folderResults + "\\" + $"FallDataAllAvgAutoShift_{dateString}.csv");
            TextWriter ts6 = new StreamWriter(folderResults + "\\" + $"FallDataAvgFixed32AutoShift_{dateString}.csv");
            TextWriter ts7 = new StreamWriter(folderResults + "\\" + $"FallDataAvgLifeModel32AutoShift_{dateString}.csv");

            TextWriter ts8 = new StreamWriter(folderResults + "\\" + $"FallDataAvgFixed32AutoShift_10Miss_{dateString}.csv");
            TextWriter ts9 = new StreamWriter(folderResults + "\\" + $"FallDataAvgLifeModel32AutoShift_10Miss_{dateString}.csv");

            TextWriter ts10 = new StreamWriter(folderResults + "\\" + $"FallDataAvgFixed32AutoShift_50Miss_{dateString}.csv");
            TextWriter ts11 = new StreamWriter(folderResults + "\\" + $"FallDataAvgLifeModel32AutoShift_50Miss_{dateString}.csv");

            foreach (var sample in samples)
            {
                ts.WriteLine(sample.ToString());
                ts2.WriteLine(sample.ToAverageString());
                ts3.WriteLine(sample.ToLifeModel(maxTimeSpan, true));
                ts4.WriteLine(sample.ToLifeModel(maxTimeSpan, false));//Life Model
                ts5.Write(sample.ToString(true));
                ts6.Write(sample.ToLifeModel(maxTimeSpan, true, Autoshift: true));
                ts7.Write(sample.ToLifeModel(maxTimeSpan, false, Autoshift: true));//Life Model

                ts8.Write(sample.ToLifeModel(maxTimeSpan, true, Autoshift: true, missingRate: 0.1, randSeed: 10));
                ts9.Write(sample.ToLifeModel(maxTimeSpan, false, Autoshift: true, missingRate: 0.1, randSeed: 10));//Life Model

                ts10.Write(sample.ToLifeModel(maxTimeSpan, true, Autoshift: true, missingRate: 0.5, randSeed: 10));
                ts11.Write(sample.ToLifeModel(maxTimeSpan, false, Autoshift: true, missingRate: 0.5, randSeed: 10));//Life Model


            }
            ts.Close();
            ts2.Close();
            ts3.Close();
            ts4.Close();
            ts5.Close();
            ts6.Close();
            ts7.Close();
            ts8.Close();
            ts9.Close();
            ts10.Close();
            ts11.Close();


            return Task.FromResult(noOfPatients);

        }
    }
}

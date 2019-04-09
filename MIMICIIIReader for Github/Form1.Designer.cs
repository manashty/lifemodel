namespace MIMICIIIReader
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.btn_Read = new System.Windows.Forms.Button();
            this.btn_Cancel = new System.Windows.Forms.Button();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.btn_GetPatient = new System.Windows.Forms.Button();
            this.richTextBox1 = new System.Windows.Forms.RichTextBox();
            this.btn_NormalizeTime = new System.Windows.Forms.Button();
            this.btn_WriteOutput = new System.Windows.Forms.Button();
            this.btn_LifeModel = new System.Windows.Forms.Button();
            this.btn_TestAPI = new System.Windows.Forms.Button();
            this.btn_ReadFallData = new System.Windows.Forms.Button();
            this.propertyGrid1 = new System.Windows.Forms.PropertyGrid();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // btn_Read
            // 
            this.btn_Read.Location = new System.Drawing.Point(5, 18);
            this.btn_Read.Margin = new System.Windows.Forms.Padding(2);
            this.btn_Read.Name = "btn_Read";
            this.btn_Read.Size = new System.Drawing.Size(106, 64);
            this.btn_Read.TabIndex = 0;
            this.btn_Read.Text = "Read";
            this.btn_Read.UseVisualStyleBackColor = true;
            this.btn_Read.Click += new System.EventHandler(this.btn_Read_Click);
            // 
            // btn_Cancel
            // 
            this.btn_Cancel.Location = new System.Drawing.Point(36, 201);
            this.btn_Cancel.Margin = new System.Windows.Forms.Padding(2);
            this.btn_Cancel.Name = "btn_Cancel";
            this.btn_Cancel.Size = new System.Drawing.Size(106, 64);
            this.btn_Cancel.TabIndex = 1;
            this.btn_Cancel.Text = "Cancel";
            this.btn_Cancel.UseVisualStyleBackColor = true;
            this.btn_Cancel.Click += new System.EventHandler(this.btn_Cancel_Click);
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // btn_GetPatient
            // 
            this.btn_GetPatient.Enabled = false;
            this.btn_GetPatient.Location = new System.Drawing.Point(135, 86);
            this.btn_GetPatient.Margin = new System.Windows.Forms.Padding(2);
            this.btn_GetPatient.Name = "btn_GetPatient";
            this.btn_GetPatient.Size = new System.Drawing.Size(106, 64);
            this.btn_GetPatient.TabIndex = 2;
            this.btn_GetPatient.Text = "Get Patient Data";
            this.btn_GetPatient.UseVisualStyleBackColor = true;
            this.btn_GetPatient.Click += new System.EventHandler(this.btn_GetPatient_Click);
            // 
            // richTextBox1
            // 
            this.richTextBox1.Dock = System.Windows.Forms.DockStyle.Right;
            this.richTextBox1.Location = new System.Drawing.Point(658, 0);
            this.richTextBox1.Name = "richTextBox1";
            this.richTextBox1.Size = new System.Drawing.Size(626, 585);
            this.richTextBox1.TabIndex = 3;
            this.richTextBox1.Text = "";
            // 
            // btn_NormalizeTime
            // 
            this.btn_NormalizeTime.Enabled = false;
            this.btn_NormalizeTime.Location = new System.Drawing.Point(135, 18);
            this.btn_NormalizeTime.Margin = new System.Windows.Forms.Padding(2);
            this.btn_NormalizeTime.Name = "btn_NormalizeTime";
            this.btn_NormalizeTime.Size = new System.Drawing.Size(106, 64);
            this.btn_NormalizeTime.TabIndex = 2;
            this.btn_NormalizeTime.Text = "Normalize Admission Time";
            this.btn_NormalizeTime.UseVisualStyleBackColor = true;
            this.btn_NormalizeTime.Click += new System.EventHandler(this.btn_NormalizeTime_Click);
            // 
            // btn_WriteOutput
            // 
            this.btn_WriteOutput.Location = new System.Drawing.Point(5, 86);
            this.btn_WriteOutput.Margin = new System.Windows.Forms.Padding(2);
            this.btn_WriteOutput.Name = "btn_WriteOutput";
            this.btn_WriteOutput.Size = new System.Drawing.Size(106, 64);
            this.btn_WriteOutput.TabIndex = 4;
            this.btn_WriteOutput.Text = "Write Output";
            this.btn_WriteOutput.UseVisualStyleBackColor = true;
            this.btn_WriteOutput.Click += new System.EventHandler(this.btn_WriteOutput_Click);
            // 
            // btn_LifeModel
            // 
            this.btn_LifeModel.Location = new System.Drawing.Point(130, 18);
            this.btn_LifeModel.Margin = new System.Windows.Forms.Padding(2);
            this.btn_LifeModel.Name = "btn_LifeModel";
            this.btn_LifeModel.Size = new System.Drawing.Size(106, 64);
            this.btn_LifeModel.TabIndex = 5;
            this.btn_LifeModel.Text = "Test Life Model by Creating a Sample";
            this.btn_LifeModel.UseVisualStyleBackColor = true;
            this.btn_LifeModel.Click += new System.EventHandler(this.btn_LifeModel_Click);
            // 
            // btn_TestAPI
            // 
            this.btn_TestAPI.Location = new System.Drawing.Point(5, 18);
            this.btn_TestAPI.Margin = new System.Windows.Forms.Padding(2);
            this.btn_TestAPI.Name = "btn_TestAPI";
            this.btn_TestAPI.Size = new System.Drawing.Size(106, 64);
            this.btn_TestAPI.TabIndex = 6;
            this.btn_TestAPI.Text = "Predict using API";
            this.btn_TestAPI.UseVisualStyleBackColor = true;
            this.btn_TestAPI.Click += new System.EventHandler(this.btn_TestAPI_Click);
            // 
            // btn_ReadFallData
            // 
            this.btn_ReadFallData.Location = new System.Drawing.Point(5, 18);
            this.btn_ReadFallData.Margin = new System.Windows.Forms.Padding(2);
            this.btn_ReadFallData.Name = "btn_ReadFallData";
            this.btn_ReadFallData.Size = new System.Drawing.Size(162, 64);
            this.btn_ReadFallData.TabIndex = 7;
            this.btn_ReadFallData.Text = "Read && Write Fall Data (Auto)";
            this.btn_ReadFallData.UseVisualStyleBackColor = true;
            this.btn_ReadFallData.Click += new System.EventHandler(this.btn_ReadFallData_Click_1);
            // 
            // propertyGrid1
            // 
            this.propertyGrid1.Location = new System.Drawing.Point(301, 51);
            this.propertyGrid1.Name = "propertyGrid1";
            this.propertyGrid1.Size = new System.Drawing.Size(351, 522);
            this.propertyGrid1.TabIndex = 8;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btn_Read);
            this.groupBox1.Controls.Add(this.btn_WriteOutput);
            this.groupBox1.Controls.Add(this.btn_NormalizeTime);
            this.groupBox1.Controls.Add(this.btn_GetPatient);
            this.groupBox1.Location = new System.Drawing.Point(36, 281);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(259, 169);
            this.groupBox1.TabIndex = 9;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Mortality Prediction";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.btn_ReadFallData);
            this.groupBox2.Location = new System.Drawing.Point(36, 51);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(259, 136);
            this.groupBox2.TabIndex = 10;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Fall Forecasting";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.btn_LifeModel);
            this.groupBox3.Controls.Add(this.btn_TestAPI);
            this.groupBox3.Location = new System.Drawing.Point(41, 456);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(254, 117);
            this.groupBox3.TabIndex = 11;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "PHARMS API Test";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1284, 585);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.propertyGrid1);
            this.Controls.Add(this.richTextBox1);
            this.Controls.Add(this.btn_Cancel);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "Form1";
            this.Text = "MIMIC III Reader";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button btn_Read;
        private System.Windows.Forms.Button btn_Cancel;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.Button btn_GetPatient;
        private System.Windows.Forms.RichTextBox richTextBox1;
        private System.Windows.Forms.Button btn_NormalizeTime;
        private System.Windows.Forms.Button btn_WriteOutput;
        private System.Windows.Forms.Button btn_LifeModel;
        private System.Windows.Forms.Button btn_TestAPI;
        private System.Windows.Forms.Button btn_ReadFallData;
        private System.Windows.Forms.PropertyGrid propertyGrid1;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupBox3;
    }
}


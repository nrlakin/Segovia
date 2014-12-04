/*
 * PsMainFrame.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */


package com.fujitsu.frontech.palmsecure_smpl;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.SourceDataLine;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingConstants;
import javax.swing.border.BevelBorder;

import javax.swing.JScrollPane;
import javax.swing.AbstractListModel;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionListener;
import javax.swing.event.ListSelectionEvent;

import com.fujitsu.frontech.palmsecure_smpl.data.PsDataManager;
import com.fujitsu.frontech.palmsecure_smpl.data.PsLogManager;
import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;
import com.fujitsu.frontech.palmsecure_smpl.event.PsBusinessListener;
import com.fujitsu.frontech.palmsecure_smpl.exception.PsAplException;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorIni;
import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsMainFrame extends JFrame implements ActionListener, WindowListener, PsBusinessListener {

	private static final long serialVersionUID = 1L;

	private static final String PS_HANDGUIDE_FILENAME = "PalmSecureSample_HANDGUIDE.bmp";
	private static final String PS_GUIDELESS_FILENAME = "PalmSecureSample_GUIDELESS.bmp";
	private static final String PS_SOUND_OK_FILENAME = "PalmSecureSample_OK.wav";
	private static final String PS_SOUND_NG_FILENAME = "PalmSecureSample_NG.wav";

	private static final String PS_FONT_NAME = null;
	private static final int PS_BTN_FONT_SIZE = 18;
	private static final int PS_BTN_WIDTH = 121;
	private static final int PS_BTN_HEIGHT = 57;

	private static final int PS_REGISTER_MAX = 1000;

	private JTextField txtSensorMode = null;
	private JTextField txtProcess = null;
	private JTextField txtIDNum = null;
	private JTextArea txtGuidance = null;
	private JLabel lblID = null;
	private JLabel lblIdList = null;
	private JLabel lblIDNum = null;
	private JList lstID = null;
	private JButton btnEnroll = null;
	private JButton btnDelete = null;
	private JButton btnVerify = null;
	private JButton btnIdentify = null;
	private JButton btnCancel = null;
	private JButton btnClose = null;
	private PsSilhouetteLabel lblImage = null;
	private PsInputIdText txtID = null;

	private long usingGuideMode = 0;
	private long usingSensorType = 0;
	private long usingSensorExtKind = 0;

	private BufferedImage guideImage = null;
	private byte[] silhouetteLog = null;

	private PsStateCallback psStateCB = null;
	private PsStreamingCallback psStreamingCB = null;

    private PalmSecureIf palmsecureIf = null;

    private JAVA_uint32 ModuleHandle = null;

	public boolean cancelFlg = false;
	public boolean enrollFlg = false;
	public int notifiedScore = 0;
    public byte[] silhouette = null;

	public PsMainFrame() {
		Font buttonFont = new Font(PS_FONT_NAME, Font.BOLD, PS_BTN_FONT_SIZE);

		// Main window
		setBounds(100, 100, 800, 550);
		setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
		getContentPane().setLayout(null);

		// Text(display):Kind of sensor and Mode of hand-guide
		txtSensorMode = new JTextField();
		txtSensorMode.setEditable(false);
		txtSensorMode.setFont(new Font(PS_FONT_NAME, Font.BOLD, 20));
		txtSensorMode.setHorizontalAlignment(SwingConstants.CENTER);
		txtSensorMode.setText("Sensor name");
		txtSensorMode.setBackground(Color.CYAN);
		txtSensorMode.setBounds(12, 10, 760, 35);
		getContentPane().add(txtSensorMode);
		txtSensorMode.setColumns(10);

		// Label:Image display
		lblImage = new PsSilhouetteLabel();
		lblImage.setBorder(null);
		lblImage.setBounds(12, 55, 369, 264);
		getContentPane().add(lblImage);

		// Panel:ID display
		JPanel pnlID = new JPanel();
		pnlID.setBorder(new BevelBorder(BevelBorder.LOWERED, null, null, null, null));
		pnlID.setBounds(393, 55, 246, 57);
		getContentPane().add(pnlID);
		pnlID.setLayout(null);

		// Label:ID
		lblID = new JLabel("ID");
		lblID.setHorizontalAlignment(SwingConstants.CENTER);
		lblID.setFont(new Font(PS_FONT_NAME, Font.PLAIN, 16));
		lblID.setBounds(12, 14, 36, 33);
		pnlID.add(lblID);

		// Text(input):ID
		txtID = new PsInputIdText();
		txtID.setFont(new Font(PS_FONT_NAME, Font.PLAIN, 16));
		txtID.setText("1234567890123456");
		txtID.setBounds(52, 14, 182, 33);
		pnlID.add(txtID);
		txtID.setColumns(10);

		// Button:Enroll
		btnEnroll = new JButton("Enroll");
		btnEnroll.setFont(buttonFont);
		btnEnroll.setBounds(651, 55, PS_BTN_WIDTH, PS_BTN_HEIGHT);
		getContentPane().add(btnEnroll);

		// Button:Delete
		btnDelete = new JButton("Delete");
		btnDelete.setFont(buttonFont);
		btnDelete.setBounds(651, 122, PS_BTN_WIDTH, PS_BTN_HEIGHT);
		getContentPane().add(btnDelete);

		// Button:Verify
		btnVerify = new JButton("Verify");
		btnVerify.setFont(buttonFont);
		btnVerify.setBounds(651, 217, PS_BTN_WIDTH, PS_BTN_HEIGHT);
		getContentPane().add(btnVerify);

		// Button:Identify
		btnIdentify = new JButton("Identify");
		btnIdentify.setFont(buttonFont);
		btnIdentify.setBounds(651, 285, PS_BTN_WIDTH, PS_BTN_HEIGHT);
		getContentPane().add(btnIdentify);

		// Button:Cancel
		btnCancel = new JButton("Cancel");
		btnCancel.setEnabled(false);
		btnCancel.setFont(buttonFont);
		btnCancel.setBounds(651, 378, PS_BTN_WIDTH, PS_BTN_HEIGHT);
		getContentPane().add(btnCancel);

		// Button:Close
		btnClose = new JButton("Close");
		btnClose.setFont(buttonFont);
		btnClose.setBounds(651, 445, PS_BTN_WIDTH, PS_BTN_HEIGHT);
		getContentPane().add(btnClose);

		// Text(display):Processing state
		txtProcess = new JTextField();
		txtProcess.setText("Process");
		txtProcess.setHorizontalAlignment(SwingConstants.LEFT);
		txtProcess.setFont(new Font(PS_FONT_NAME, Font.PLAIN, 20));
		txtProcess.setEditable(false);
		txtProcess.setColumns(10);
		txtProcess.setBackground(Color.GREEN);
		txtProcess.setBounds(12, 329, 369, 35);
		getContentPane().add(txtProcess);

		// Text(display):Guidance
		txtGuidance = new JTextArea();
		txtGuidance.setEditable(false);
		txtGuidance.setBorder(new BevelBorder(BevelBorder.LOWERED, null, null, null, null));
		txtGuidance.setFont(new Font(PS_FONT_NAME, Font.BOLD, 20));
		txtGuidance.setForeground(Color.BLUE);
		txtGuidance.setText("Guidance");
		txtGuidance.setBounds(12, 374, 627, 128);
		getContentPane().add(txtGuidance);

		// Panel:ID-list
		JPanel pnlIDList = new JPanel();
		pnlIDList.setBorder(new BevelBorder(BevelBorder.LOWERED, null, null, null, null));
		pnlIDList.setBounds(393, 122, 246, 242);
		getContentPane().add(pnlIDList);
		pnlIDList.setLayout(null);

		// Text(display):number of ID
		txtIDNum = new JTextField();
		txtIDNum.setEditable(false);
		txtIDNum.setHorizontalAlignment(SwingConstants.RIGHT);
		txtIDNum.setText("1000");
		txtIDNum.setFont(new Font(PS_FONT_NAME, Font.PLAIN, 16));
		txtIDNum.setColumns(10);
		txtIDNum.setBounds(185, 10, 49, 26);
		pnlIDList.add(txtIDNum);

		// Label:ID-list
		lblIdList = new JLabel("ID List");
		lblIdList.setFont(new Font("MS UI Gothic", Font.PLAIN, 14));
		lblIdList.setBounds(12, 20, 68, 26);
		pnlIDList.add(lblIdList);

		// label:number of ID
		lblIDNum = new JLabel("ID num");
		lblIDNum.setHorizontalAlignment(SwingConstants.RIGHT);
		lblIDNum.setFont(new Font("MS UI Gothic", Font.PLAIN, 14));
		lblIDNum.setBounds(74, 9, 100, 26);
		pnlIDList.add(lblIDNum);

		// list:ID
		JScrollPane scrollPane = new JScrollPane();
		scrollPane.setBounds(12, 46, 222, 186);
		pnlIDList.add(scrollPane);
		lstID = new JList();
		lstID.addListSelectionListener(new ListSelectionListener() {
			public void valueChanged(ListSelectionEvent arg0) {
				String selectID = (String) lstID.getSelectedValue();
				txtID.setText(selectID);
			}
		});
		lstID.addMouseListener(new MouseListener() {
			public void mouseClicked(MouseEvent e) {
				String selectID = (String) lstID.getSelectedValue();
				txtID.setText(selectID);
			}
			public void mousePressed(MouseEvent e) {
			}
			public void mouseReleased(MouseEvent e) {
			}
			public void mouseEntered(MouseEvent e) {
			}
			public void mouseExited(MouseEvent e) {
			}
		});
		lstID.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		lstID.setFont(new Font("MS UI Gothic", Font.PLAIN, 16));
		lstID.setModel(new AbstractListModel() {
			private static final long serialVersionUID = 1L;
			String[] values = new String[] {"1234567890123456", "id2"};
			public int getSize() {
				return values.length;
			}
			public Object getElementAt(int index) {
				return values[index];
			}
		});
		scrollPane.setViewportView(lstID);
	}

	public boolean Ps_Sample_Apl_Java() {

		File checkFile = null;

		checkFile = new File(PsFileAccessorLang.FileName);
		if (!checkFile.exists()) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PsFileAccessorLang.FileName,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		checkFile = new File(PsFileAccessorIni.FileName);
		if (!checkFile.exists()) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PsFileAccessorIni.FileName,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		checkFile = new File(PS_GUIDELESS_FILENAME);
		if (!checkFile.exists()) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PS_GUIDELESS_FILENAME,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		checkFile = new File(PS_HANDGUIDE_FILENAME);
		if (!checkFile.exists()) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PS_HANDGUIDE_FILENAME,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		checkFile = new File(PS_SOUND_OK_FILENAME);
		if (!checkFile.exists()) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PS_SOUND_OK_FILENAME,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		checkFile = new File(PS_SOUND_NG_FILENAME);
		if (!checkFile.exists()) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PS_SOUND_NG_FILENAME,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		// Read the language file.
		PsFileAccessorLang psFileAcsLang = PsFileAccessorLang.GetInstance();
		if (psFileAcsLang == null) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PsFileAccessorLang.FileName,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		// Read the Configuration file.
		PsFileAccessorIni psFileAcsIni = PsFileAccessorIni.GetInstance();
		if (psFileAcsIni == null) {
			JOptionPane.showMessageDialog(
					null,
					"Read Error " + PsFileAccessorIni.FileName,
					null,
					JOptionPane.ERROR_MESSAGE);
			System.exit(0);
		}

		// Initialize the Authentication library.
		if (!Ps_Sample_Apl_Java_InitLibrary()) {
			System.exit(0);
		}

		if (usingSensorType == PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_TYPE_4) {
			PsFileAccessorLang.GetInstance().SetValue(
					PsFileAccessorLang.Guidance_WorkEnroll,
					PsFileAccessorLang.GetInstance().GetValue(PsFileAccessorLang.Guidance_WorkEnroll3));
			if (usingSensorExtKind == PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_MODE_EXTEND) {
				usingSensorType = PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_TYPE_8;
				PsFileAccessorLang.GetInstance().SetValue(
						PsFileAccessorLang.MainDialog_SensorName7,
						PsFileAccessorLang.GetInstance().GetValue(PsFileAccessorLang.MainDialog_SensorName3));
			}
		} else if (usingSensorType == PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_TYPE_2) {
			if (usingSensorExtKind == PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_MODE_COMPATIBLE) {
				PsFileAccessorLang.GetInstance().SetValue(
						PsFileAccessorLang.MainDialog_SensorName1,
						PsFileAccessorLang.GetInstance().GetValue(PsFileAccessorLang.MainDialog_SensorName1_Compati));
			} else if (usingSensorExtKind == PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_MODE_EXTEND){
				PsFileAccessorLang.GetInstance().SetValue(
						PsFileAccessorLang.MainDialog_SensorName1,
						PsFileAccessorLang.GetInstance().GetValue(PsFileAccessorLang.MainDialog_SensorName1_Extend));
			}
		} else if (usingSensorType == PalmSecureConstant.JAVA_PvAPI_INFO_SENSOR_TYPE_8) {
			PsFileAccessorLang.GetInstance().SetValue(
					PsFileAccessorLang.Guidance_WorkEnroll,
					PsFileAccessorLang.GetInstance().GetValue(PsFileAccessorLang.Guidance_WorkEnroll3));
		}



		try {
			Ps_Sample_Apl_Java_ReadGuideBmp();
			Ps_Sample_Apl_Java_InitControls();
			Ps_Sample_Apl_Java_InitIdList();
		} catch(PsAplException pae) {
			JOptionPane.showMessageDialog(null, "Initialize Error");
			System.exit(0);
		}

		return true;
	}

	/**
	 * Initialize the Authentication library.
	 */
	private boolean Ps_Sample_Apl_Java_InitLibrary()
	{

		long ret = PalmSecureConstant.JAVA_BioAPI_FALSE;

		String Key = PsFileAccessorIni.GetInstance().GetValueString(
						PsFileAccessorIni.ApplicationKey);

		byte[] ModuleGuid = new byte[]{
				(byte)0xe1, (byte)0x9a, (byte)0x69, (byte)0x01,
				(byte)0xb8, (byte)0xc2, (byte)0x49, (byte)0x80,
				(byte)0x87, (byte)0x7e, (byte)0x11, (byte)0xd4,
				(byte)0xd8, (byte)0xf1, (byte)0xbe, (byte)0x79
		};

		JAVA_uint32 lpuiSensorNum = new JAVA_uint32();
		JAVA_PvAPI_SensorInfoEx[] lptSensorInfo =
			new JAVA_PvAPI_SensorInfoEx[(int)PalmSecureConstant.JAVA_PvAPI_GET_SENSOR_INFO_MAX];

		JAVA_PvAPI_LBINFO lbInfo = new JAVA_PvAPI_LBINFO();

		JAVA_PvAPI_ErrorInfo errorInfo = new JAVA_PvAPI_ErrorInfo();

		//Create a instance of PalmSecureIf class
		///////////////////////////////////////////////////////////////////////////
		palmsecureIf = new PalmSecureIf();
		///////////////////////////////////////////////////////////////////////////

		//Authenticate application by key
		///////////////////////////////////////////////////////////////////////////
		try {
			ret = palmsecureIf.JAVA_PvAPI_ApAuthenticate(Key);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////


		//Load module
		///////////////////////////////////////////////////////////////////////////
		try {
			ret = palmsecureIf.JAVA_BioAPI_ModuleLoad(ModuleGuid, null, null, null);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////


		//Get connected sensor information
		///////////////////////////////////////////////////////////////////////////
		try {
			ret = palmsecureIf.JAVA_PvAPI_GetConnectSensorInfoEx(
					lpuiSensorNum,
					lptSensorInfo);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////


		//Attatch to module
		///////////////////////////////////////////////////////////////////////////
		try {
			ModuleHandle = new JAVA_uint32();
			ret = palmsecureIf.JAVA_BioAPI_ModuleAttach(
					ModuleGuid,
					null,
					null,
					null,
					null,
					null,
					null,
					null,
					null,
					null,
					ModuleHandle);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////


		//Set action listener
		///////////////////////////////////////////////////////////////////////////
		try {
			psStreamingCB = new PsStreamingCallback();
			psStateCB = new PsStateCallback();
			ret = palmsecureIf.JAVA_BioAPI_SetGUICallbacks(
					ModuleHandle,
					psStreamingCB,
					this,
					psStateCB,
					this);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////


		//Get library information
		///////////////////////////////////////////////////////////////////////////
		try {
			ret = palmsecureIf.JAVA_PvAPI_GetLibraryInfo(lbInfo);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////

		usingSensorType = (int)lptSensorInfo[0].uiSensor;
		usingSensorExtKind = lbInfo.uiSensorExtKind;
		usingGuideMode = PsFileAccessorIni.GetInstance().GetValueInteger(PsFileAccessorIni.GuideMode);

		//Set guide mode
		///////////////////////////////////////////////////////////////////////////
		JAVA_uint32 dwFlag = new JAVA_uint32();
		dwFlag.value = PalmSecureConstant.JAVA_PvAPI_PROFILE_GUIDE_MODE;
		JAVA_uint32 dwParam1 = new JAVA_uint32();
		if ( usingGuideMode == 1 ) {
			dwParam1.value = (int)PalmSecureConstant.JAVA_PvAPI_PROFILE_GUIDE_MODE_GUIDE;
		} else {
			dwParam1.value = (int)PalmSecureConstant.JAVA_PvAPI_PROFILE_GUIDE_MODE_NO_GUIDE;
		}
		try {
			ret = palmsecureIf.JAVA_PvAPI_SetProfile(
					ModuleHandle,
					dwFlag,
					dwParam1,
					null,
					null);
			if (ret != PalmSecureConstant.JAVA_BioAPI_OK) {
				palmsecureIf.JAVA_PvAPI_GetErrorInfo(errorInfo);
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, errorInfo);
				return false;
			}
		} catch(PalmSecureException e) {
			PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, e);
			return false;
		}
		///////////////////////////////////////////////////////////////////////////

		return true;
	}

	/**
	 * Terminate the Authentication library.
	 */
	private void Ps_Sample_Apl_Java_TermLibrary() {

		long ret = PalmSecureConstant.JAVA_BioAPI_FALSE;

		byte[] ModuleGuid = new byte[]{
				(byte)0xe1, (byte)0x9a, (byte)0x69, (byte)0x01,
				(byte)0xb8, (byte)0xc2, (byte)0x49, (byte)0x80,
				(byte)0x87, (byte)0x7e, (byte)0x11, (byte)0xd4,
				(byte)0xd8, (byte)0xf1, (byte)0xbe, (byte)0x79
		};

		//Detach module
		///////////////////////////////////////////////////////////////////////////
		try {
			ret = palmsecureIf.JAVA_BioAPI_ModuleDetach(
					ModuleHandle);
		} catch(PalmSecureException e) {
		}
		///////////////////////////////////////////////////////////////////////////


		//Unload module
		///////////////////////////////////////////////////////////////////////////
		try {
			ret = palmsecureIf.JAVA_BioAPI_ModuleUnload(
					ModuleGuid,
					null,
					null);
		} catch(PalmSecureException e) {
		}
		///////////////////////////////////////////////////////////////////////////

		return;

	}

	public void Ps_Sample_Apl_Java_SetSilhouette(JAVA_BioAPI_GUI_BITMAP Bitmap) {

		byte[] silhouette = Bitmap.Bitmap.Data;
		if ((silhouette != null) && (0 < silhouette.length)) {

			try {
				// Display the silhouette image.
				lblImage.Ps_Sample_Apl_Java_SetSilhouette(silhouette);
				this.silhouette = silhouette;
				lblImage.repaint();

				silhouetteLog = silhouette;
			}
			catch (IOException ioEx) {
			}
		}
	}

	public byte[] Ps_Sample_Apl_Java_GetSilhouette() {

		return this.silhouette;
	}

	public void Ps_Sample_Apl_Java_PlayWave(String fileName) {

		try {
			File soundFile = new File(fileName);
			AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(soundFile);
			AudioFormat audioFormat = audioInputStream.getFormat();
			DataLine.Info info = new DataLine.Info(SourceDataLine.class, audioFormat);
			SourceDataLine line = (SourceDataLine) AudioSystem.getLine(info);
			line.open(audioFormat);
			line.start();

			int nBytesRead = 0;
			byte[] abData = new byte[128000];
			while (nBytesRead != -1) {

				nBytesRead = audioInputStream.read(abData, 0, abData.length);
				if (nBytesRead >= 0) {
					line.write(abData, 0, nBytesRead);
				}
			}

			line.drain();
			line.close();
		}
		catch (Exception e) {
		}
	}


	public void Ps_Sample_Apl_Java_SetSilhouetteLog(byte[] silhouetteLog) {
		this.silhouetteLog = silhouetteLog;
	}


	public byte[] Ps_Sample_Apl_Java_GetSilhouetteLog() {
		return silhouetteLog;
	}


	private void Ps_Sample_Apl_Java_ReadGuideBmp() throws PsAplException {

		guideImage = null;

		File guideFile;
		if ( usingGuideMode == 0 ) {
			guideFile = new File(PS_GUIDELESS_FILENAME);
		} else {
			guideFile = new File(PS_HANDGUIDE_FILENAME);
		}

		try {
			guideImage = ImageIO.read(guideFile);
		} catch (IOException e) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorGuideBmpLoad);
			throw pae;
		}

		if (guideImage == null) {
			PsAplException pae = new PsAplException(PsFileAccessorLang.ErrorMessage_AplErrorGuideBmpLoad);
			throw pae;
		}
	}


	private void Ps_Sample_Apl_Java_DisplayGuideBmp() {

		if (guideImage != null) {
			lblImage.Ps_Sample_Apl_Java_SetHandGuide(guideImage);
			lblImage.repaint();
		}
	}

	private void Ps_Sample_Apl_Java_SetGuidance(String guidanceKey, boolean error) {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		String guidance;
		if (guidanceKey.compareTo("") == 0) {
			guidance = "";
		} else {
			guidance = langAcs.GetValue(guidanceKey);
		}
		txtGuidance.setText(guidance);

		if (error) {
			txtGuidance.setForeground(Color.RED);
		} else {
			txtGuidance.setForeground(Color.BLUE);
		}
	}


	private void Ps_Sample_Apl_Java_SetGuidance(String guidanceKey, boolean error, String userID) {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		String guidance;
		if (guidanceKey.compareTo("") == 0) {
			guidance = "";
		} else {
			guidance = langAcs.GetValue(guidanceKey).replaceAll("\\{0\\}", userID);
		}
		txtGuidance.setText(guidance);

		if (error) {
			txtGuidance.setForeground(Color.RED);
		} else {
			txtGuidance.setForeground(Color.BLUE);
		}
	}

	private void Ps_Sample_Apl_Java_SetWorkMessage(String msgKey) {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		if (msgKey.compareTo("") == 0) {
			txtProcess.setText("");
			txtProcess.setBackground(Color.WHITE);
		} else {
			txtProcess.setText(langAcs.GetValue(msgKey));
			txtProcess.setBackground(Color.GREEN);
		}
	}

	private void Ps_Sample_Apl_Java_SetWorkMessage(String msgKey, int count) {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		if (msgKey.compareTo("") == 0) {
			txtProcess.setText("");
			txtProcess.setBackground(Color.WHITE);
		} else {
			String workMessage = langAcs.GetValue(msgKey).replaceAll("\\{0\\}", String.valueOf(count));
			txtProcess.setText(workMessage);
			txtProcess.setBackground(Color.GREEN);
		}
	}

	private void Ps_Sample_Apl_Java_InitControls() {

		PsFileAccessorLang langAcs = PsFileAccessorLang.GetInstance();

		// Main window.
		String title = langAcs.GetValue(PsFileAccessorLang.MainDialog_Title);
		setTitle(title);
		Dimension d = Toolkit.getDefaultToolkit().getScreenSize();
		setLocation((d.width - getSize().width) / 2, (d.height - getSize().height) / 2);
		setResizable(false);

		// Text(display):Kind of sensor and Mode of hand-guide
		String sensorMode = langAcs.GetValue(PsFileAccessorLang.MainDialog_SensorName + String.valueOf(usingSensorType));
		String guide = null;
		if ( usingGuideMode == 0 ) {
			guide = langAcs.GetValue(PsFileAccessorLang.MainDialog_GuideMode0);
		} else {
			guide = langAcs.GetValue(PsFileAccessorLang.MainDialog_GuideMode1);
		}
		sensorMode += " " + guide;
		txtSensorMode.setText(sensorMode);

		// Label:Image display
		lblImage.setText("");
		Ps_Sample_Apl_Java_DisplayGuideBmp();

		// Label:ID
		String idLblTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_IdLbl);
		lblID.setText(idLblTxt);

		// Text(input):ID
		txtID.setText("");

		// Button:Enroll
		String enrollBtnTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_EnrollBtn);
		btnEnroll.setText(enrollBtnTxt);
		btnEnroll.addActionListener(this);

		// Button:Delete
		String deleteBtnTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_DeleteBtn);
		btnDelete.setText(deleteBtnTxt);
		btnDelete.addActionListener(this);

		// Button:Verify
		String verifyBtnTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_VerifyBtn);
		btnVerify.setText(verifyBtnTxt);
		btnVerify.addActionListener(this);

		// Button:Identify
		String identifyBtnTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_IdentifyBtn);
		btnIdentify.setText(identifyBtnTxt);
		btnIdentify.addActionListener(this);

		// Button:Cancel
		String cancelBtnTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_CancelBtn);
		btnCancel.setText(cancelBtnTxt);
		btnCancel.addActionListener(this);

		// Button:Close
		String exitBtnTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_ExitBtn);
		btnClose.setText(exitBtnTxt);
		btnClose.addActionListener(this);

		// Text(display):Processing state
		txtProcess.setText("");
		txtProcess.setBackground(Color.WHITE);

		// Text(display):Guidance
		txtGuidance.setText("");
		txtGuidance.setOpaque(true);
		txtGuidance.setForeground(Color.BLUE);
		txtGuidance.setBackground(Color.WHITE);
		txtGuidance.setLineWrap(true);

		// Text(display):number of ID
		txtIDNum.setText("");

		// Label:ID-list
		String idListTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_IdListLbl);
		lblIdList.setText(idListTxt);

		// Label:number of ID
		String idNumTxt = langAcs.GetValue(PsFileAccessorLang.MainDialog_IdNumLbl);
		lblIDNum.setText(idNumTxt);

		addWindowListener(this);
	}

	private void Ps_Sample_Apl_Java_InitIdList() throws PsAplException {
		PsDataManager dataMng = PsDataManager.GetInstance();

		dataMng.Ps_Sample_Apl_Java_Init(usingSensorType, usingGuideMode);
		Ps_Sample_Apl_Java_UpdateIdList();
	}

	private void Ps_Sample_Apl_Java_UpdateIdList() {
		PsDataManager dataMng = PsDataManager.GetInstance();

		ArrayList<String> idList = dataMng.Ps_Sample_Apl_Java_GetAllUserId();
		lstID.setListData(idList.toArray());

		Integer intNum = new Integer(idList.size());
		txtIDNum.setText(intNum.toString());
	}

	public void Ps_Sample_Apl_Java_InitState() {
		Ps_Sample_Apl_Java_SetComponentEnabled(true);
		txtID.requestFocus();
		txtID.selectAll();
		cancelFlg = false;
	}

	private void Ps_Sample_Apl_Java_SetComponentEnabled(boolean enabled) {

		if (enabled) {
			btnEnroll.setEnabled(true);
			btnDelete.setEnabled(true);
			btnVerify.setEnabled(true);
			btnIdentify.setEnabled(true);
			btnCancel.setEnabled(false);
			btnClose.setEnabled(true);
			txtID.setEnabled(true);
		} else {
			btnEnroll.setEnabled(false);
			btnDelete.setEnabled(false);
			btnVerify.setEnabled(false);
			btnIdentify.setEnabled(false);
			btnCancel.setEnabled(true);
			btnClose.setEnabled(false);
			txtID.setEnabled(false);
		}
	}

	private void Ps_Sample_Apl_Java_ActionEnroll() {

		PsDataManager dataMng = PsDataManager.GetInstance();

		Ps_Sample_Apl_Java_SetWorkMessage(PsFileAccessorLang.Guidance_WorkEnrollStart);
		Ps_Sample_Apl_Java_SetGuidance("", false);
		Ps_Sample_Apl_Java_DisplayGuideBmp();

		String id = txtID.getText();

		if (id.compareTo("") == 0) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_IllegalId, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			Ps_Sample_Apl_Java_InitState();
			return;
		}

		if (dataMng.Ps_Sample_Apl_Java_IsRegist(id)) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_RegistId, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			Ps_Sample_Apl_Java_InitState();
			return;
		}

		if (dataMng.Ps_Sample_Apl_Java_GetRegistNum() >= PS_REGISTER_MAX) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_MaxOver, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			Ps_Sample_Apl_Java_InitState();
			return;
		}

		Ps_Sample_Apl_Java_SetComponentEnabled(false);

		notifiedScore = PalmSecureConstant.JAVA_PvAPI_REGIST_SCORE_QUALITY_1;
		PsThreadEnroll enrollThread = new PsThreadEnroll(this, this, palmsecureIf, ModuleHandle, id);
		enrollThread.start();

		btnCancel.requestFocus();
	}

	private void Ps_Sample_Apl_Java_ActionDelete() {

		PsDataManager dataMng = PsDataManager.GetInstance();

		Ps_Sample_Apl_Java_SetWorkMessage(PsFileAccessorLang.Guidance_WorkDelete);
		Ps_Sample_Apl_Java_SetGuidance("", false);
		String id = txtID.getText();

		if (id.compareTo("") == 0) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_IllegalId, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			return;
		}

		if (!dataMng.Ps_Sample_Apl_Java_IsRegist(id)) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_UnregistId, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			return;
		}

		Ps_Sample_Apl_Java_SetComponentEnabled(false);
		btnCancel.setEnabled(false);

		int action = PsMessageDialog.Ps_Sample_Apl_Java_ShowConfirmDialog(this, PsFileAccessorLang.ConfirmMessage_Delete);
		if (action == JOptionPane.CANCEL_OPTION) {
			Ps_Sample_Apl_Java_InitState();
			Ps_Sample_Apl_Java_SetWorkMessage("");
			return;
		}

		try {
			dataMng.Ps_Sample_Apl_Java_Delete(id);
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.CompleteMessage_Delete, false);
		}
		catch (PsAplException ex) {
			Ps_Sample_Apl_Java_SetGuidance(ex.getErrorMsgKey(), true);
		}

		try {
			Ps_Sample_Apl_Java_InitState();
			Ps_Sample_Apl_Java_InitIdList();
			Ps_Sample_Apl_Java_SetWorkMessage("");
		} catch(PsAplException pae) {

		}
	}

	private void Ps_Sample_Apl_Java_ActionVerify() {

		PsDataManager dataMng = PsDataManager.GetInstance();

		Ps_Sample_Apl_Java_SetWorkMessage(PsFileAccessorLang.Guidance_WorkVerifyStart);
		Ps_Sample_Apl_Java_SetGuidance("", false);
		Ps_Sample_Apl_Java_DisplayGuideBmp();

		String id = txtID.getText();

		if (id.compareTo("") == 0) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_IllegalId, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			return;
		}

		if (!dataMng.Ps_Sample_Apl_Java_IsRegist(id)) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_UnregistId, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			return;
		}

		// Execute verification process.
		Ps_Sample_Apl_Java_SetComponentEnabled(false);

		PsThreadVerify verifyThread = new PsThreadVerify(this, this, palmsecureIf, ModuleHandle, id);
		verifyThread.start();

		btnCancel.requestFocus();
	}

	private void Ps_Sample_Apl_Java_ActionIdentify() {

		PsDataManager dataMng = PsDataManager.GetInstance();

		Ps_Sample_Apl_Java_SetWorkMessage(PsFileAccessorLang.Guidance_WorkIdentifyStart);
		Ps_Sample_Apl_Java_SetGuidance("", false);
		Ps_Sample_Apl_Java_DisplayGuideBmp();

		if (dataMng.Ps_Sample_Apl_Java_GetRegistNum() <= 0) {
			Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.Guidance_Unregist, true);
			Ps_Sample_Apl_Java_SetWorkMessage("");
			return;
		}

		// Execute identification process.
		Ps_Sample_Apl_Java_SetComponentEnabled(false);

		PsThreadIdentify identifyThread = new PsThreadIdentify(this, this, palmsecureIf, ModuleHandle);
		identifyThread.start();

		btnCancel.requestFocus();
	}

	private void Ps_Sample_Apl_Java_ActionCancel() {

		PsThreadCancel cancelThread = new PsThreadCancel(this, this, palmsecureIf, ModuleHandle);
		cancelThread.start();
		cancelFlg = true;
	}

	private void Ps_Sample_Apl_Java_ActionExit() {
		Ps_Sample_Apl_Java_TermLibrary();
		System.exit(0);
	}

	public void Ps_Sample_Apl_Java_NotifyWorkMessage(String processKey) {

		Ps_Sample_Apl_Java_SetWorkMessage(processKey);
	}

	public void Ps_Sample_Apl_Java_NotifyWorkMessage(String processKey, int count) {

		Ps_Sample_Apl_Java_SetWorkMessage(processKey, count);
	}

	public void Ps_Sample_Apl_Java_NotifyGuidance(String guidanceKey, boolean error) {

		Ps_Sample_Apl_Java_SetGuidance(guidanceKey, error);
	}

	public void Ps_Sample_Apl_Java_NotifyResult_Enroll(PsThreadResult stResult, int enrollscore) {

		PsFileAccessorIni iniAcs = PsFileAccessorIni.GetInstance();
		PsLogManager logMng = PsLogManager.GetInstance();

		boolean enrollResult = false;

		//Show a guidance message.
		if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK
				|| stResult.authenticated == false
				|| cancelFlg == true) {

			if (cancelFlg == true) {
				Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.CompleteMessage_EnrollCancel, false);
			} else {
				Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.CompleteMessage_EnrollNg, true);
			}

		} else {
			enrollResult = true;
			if (enrollscore > PalmSecureConstant.JAVA_PvAPI_REGIST_SCORE_QUALITY_2
					|| stResult.farAchieved.get(0)  < 3000) {
				Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.CompleteMessage_EnrollRetry, false);
			} else {
				Ps_Sample_Apl_Java_SetGuidance(PsFileAccessorLang.CompleteMessage_EnrollOk, false);
			}
		}

		//Output a silhouette image
		String silhouetteFile = new String();
		if (iniAcs.GetValueInteger(PsFileAccessorIni.SilhouetteMode) == 1
				&& stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
				&& stResult.info != null
				&& cancelFlg != true)
		{
			try {
				silhouetteFile = logMng.Ps_Sample_Apl_Java_OutputSilhouette(
						iniAcs.GetValueString(PsFileAccessorIni.LogFolderPath),
						usingSensorType,
						usingGuideMode,
						"E",
						stResult.info);
			} catch (PsAplException pae) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, pae.getErrorMsgKey());
			}
        }

		if (iniAcs.GetValueInteger(PsFileAccessorIni.LogMode) == 1
				&& stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
				&& cancelFlg != true)
		{
			// Output log
			try {
				logMng.Ps_Sample_Apl_Java_WriteLog(
						iniAcs.GetValueString(PsFileAccessorIni.LogFolderPath),
						usingSensorType,
						usingGuideMode,
						"E",
						enrollResult,
						stResult.retryCnt,
						silhouetteFile,
						stResult.userId,
						stResult.farAchieved);
			} catch (PsAplException pae) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, pae.getErrorMsgKey());
			}
		}

		try {
			Ps_Sample_Apl_Java_InitState();
			Ps_Sample_Apl_Java_InitIdList();
			Ps_Sample_Apl_Java_SetWorkMessage("");
		} catch(PsAplException pae) {
		}

		return;
	}

	public void Ps_Sample_Apl_Java_NotifyResult_Verify(PsThreadResult stResult) {

		PsFileAccessorIni iniAcs = PsFileAccessorIni.GetInstance();
		PsLogManager logMng = PsLogManager.GetInstance();

		boolean verifyResult = false;

		//Show a guidance message.
		try {
			if (stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
					&& stResult.authenticated == true
					&& cancelFlg != true) {
				verifyResult = true;
				Ps_Sample_Apl_Java_PlayWave(PS_SOUND_OK_FILENAME);
				Ps_Sample_Apl_Java_SetGuidance(
						PsFileAccessorLang.CompleteMessage_VerifyOk,
						false,
						stResult.userId.get(0));
			} else {
				if (cancelFlg == true) {
					Ps_Sample_Apl_Java_SetGuidance(
							PsFileAccessorLang.CompleteMessage_VerifyCancel, false);
				} else {
					Ps_Sample_Apl_Java_PlayWave(PS_SOUND_NG_FILENAME);
					Ps_Sample_Apl_Java_SetGuidance(
							PsFileAccessorLang.CompleteMessage_VerifyNg,
							true,
							stResult.userId.get(0));
				}
			}
		} catch(Exception e) {

		}

		//Output a silhouette image
		String silhouetteFile = new String();
		if (iniAcs.GetValueInteger(PsFileAccessorIni.SilhouetteMode) == 1
				&& stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
				&& stResult.info != null
				&& cancelFlg != true) {
			try {
				silhouetteFile = logMng.Ps_Sample_Apl_Java_OutputSilhouette(
						iniAcs.GetValueString(PsFileAccessorIni.LogFolderPath),
						usingSensorType,
						usingGuideMode,
						"V",
						stResult.info);
			} catch (PsAplException pae) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, pae.getErrorMsgKey());
			}
		}

		//Output log messages
		if (iniAcs.GetValueInteger(PsFileAccessorIni.LogMode) == 1
				&& stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
				&& cancelFlg != true) {

			ArrayList<String> idList = new ArrayList<String>();
			idList.add(txtID.getText());

			// Output log
			try {
				logMng.Ps_Sample_Apl_Java_WriteLog(
						iniAcs.GetValueString(PsFileAccessorIni.LogFolderPath),
						usingSensorType,
						usingGuideMode,
						"V",
						verifyResult,
						stResult.retryCnt,
						silhouetteFile,
						stResult.userId,
						stResult.farAchieved);
			} catch (PsAplException pae) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, pae.getErrorMsgKey());
			}
		}

		Ps_Sample_Apl_Java_InitState();
		Ps_Sample_Apl_Java_SetWorkMessage("");

		return;
	}

	public void Ps_Sample_Apl_Java_NotifyResult_Identify(PsThreadResult stResult) {

		PsFileAccessorIni iniAcs = PsFileAccessorIni.GetInstance();
		PsLogManager logMng = PsLogManager.GetInstance();

		boolean identifyResult = false;

		//Show a guidance message.
		try {
			if (stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
					&& stResult.farAchieved.size() > 0
					&& stResult.authenticated == true
					&& cancelFlg != true) {

				identifyResult = true;
				Ps_Sample_Apl_Java_PlayWave(PS_SOUND_OK_FILENAME);
				Ps_Sample_Apl_Java_SetGuidance(
						PsFileAccessorLang.CompleteMessage_IdentifyOk,
						false,
						stResult.userId.get(0));
			} else {
				if (cancelFlg == true) {
					Ps_Sample_Apl_Java_SetGuidance(
							PsFileAccessorLang.CompleteMessage_IdentifyCancel, false);
				} else {
					Ps_Sample_Apl_Java_PlayWave(PS_SOUND_NG_FILENAME);
					Ps_Sample_Apl_Java_SetGuidance(
							PsFileAccessorLang.CompleteMessage_IdentifyNg, true);
				}
			}
		} catch(Exception e) {
		}

		//Output a silhouette image
		String silhouetteFile = new String();
		if (iniAcs.GetValueInteger(PsFileAccessorIni.SilhouetteMode) == 1
				&& stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
				&& stResult.info != null
				&& cancelFlg != true ) {

			try {
				silhouetteFile = logMng.Ps_Sample_Apl_Java_OutputSilhouette(
						iniAcs.GetValueString(PsFileAccessorIni.LogFolderPath),
						usingSensorType,
						usingGuideMode,
						"V",
						stResult.info);
			} catch (PsAplException pae) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, pae.getErrorMsgKey());
			}
		}

		//Output log messages
		if (iniAcs.GetValueInteger(PsFileAccessorIni.LogMode) == 1
				&& stResult.result == PalmSecureConstant.JAVA_BioAPI_OK
				&& cancelFlg != true) {

			ArrayList<String> idList = new ArrayList<String>();
			idList.add(txtID.getText());

			// Output log
			try {

				logMng.Ps_Sample_Apl_Java_WriteLog(
						iniAcs.GetValueString(PsFileAccessorIni.LogFolderPath),
						usingSensorType,
						usingGuideMode,
						"V",
						identifyResult,
						stResult.retryCnt,
						silhouetteFile,
						stResult.userId,
						stResult.farAchieved);
			} catch (PsAplException pae) {
				PsMessageDialog.Ps_Sample_Apl_Java_ShowErrorDialog(this, pae.getErrorMsgKey());
			}
		}

		Ps_Sample_Apl_Java_InitState();
		Ps_Sample_Apl_Java_SetWorkMessage("");

		return;

	}

	public void Ps_Sample_Apl_Java_NotifyResult_Cancel(PsThreadResult pvsResult) {

	}

	public void actionPerformed(ActionEvent e) {

		Object obj = e.getSource();

		if (obj.equals(btnEnroll)) {
			Ps_Sample_Apl_Java_ActionEnroll();
		}
		else if (obj.equals(btnDelete)) {
			Ps_Sample_Apl_Java_ActionDelete();
		}
		else if (obj.equals(btnVerify)) {
			Ps_Sample_Apl_Java_ActionVerify();
		}
		else if (obj.equals(btnIdentify)) {
			Ps_Sample_Apl_Java_ActionIdentify();
		}
		else if (obj.equals(btnCancel)) {
			Ps_Sample_Apl_Java_ActionCancel();
		}
		else if (obj.equals(btnClose)) {
			Ps_Sample_Apl_Java_ActionExit();
		}
	}

	public void windowActivated(WindowEvent e) {
	}

	public void windowClosed(WindowEvent e) {
	}

	public void windowClosing(WindowEvent e) {
		Ps_Sample_Apl_Java_ActionExit();
	}

	public void windowDeactivated(WindowEvent e) {
	}

	public void windowDeiconified(WindowEvent e) {
	}

	public void windowIconified(WindowEvent e) {
	}

	public void windowOpened(WindowEvent e) {
	}

}

/*
 * PsStateCallback.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import com.fujitsu.frontech.palmsecure_smpl.xml.PsFileAccessorLang;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsStateCallback implements JAVA_BioAPI_GUI_STATE_CALLBACK_IF {

	public long JAVA_BioAPI_GUI_STATE_CALLBACK(Object GuiStateCallbackCtx,
			long GuiState, short Response, long Message, short Progress,
			JAVA_BioAPI_GUI_BITMAP SampleBuffer) {

		PsMainFrame frame = (PsMainFrame) GuiStateCallbackCtx;

		if ((GuiState & PalmSecureConstant.JAVA_BioAPI_SAMPLE_AVAILABLE) ==
			PalmSecureConstant.JAVA_BioAPI_SAMPLE_AVAILABLE) {

			frame.Ps_Sample_Apl_Java_SetSilhouette(SampleBuffer);
		}

		if ((GuiState & PalmSecureConstant.JAVA_BioAPI_MESSAGE_PROVIDED) ==
			PalmSecureConstant.JAVA_BioAPI_MESSAGE_PROVIDED) {

			//Get template quality
			if ((Message & 0xff000000) == PalmSecureConstant.JAVA_PvAPI_NOTIFY_REGIST_SCORE) {
				frame.notifiedScore = (int)(Message & 0x0000000f);
				return PalmSecureConstant.JAVA_BioAPI_OK;
			}

			//Get number of capture
			if ((Message & 0xffffff00) == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_START) {

				if (frame.enrollFlg == true) {
					frame.Ps_Sample_Apl_Java_NotifyWorkMessage(
							PsFileAccessorLang.Guidance_WorkEnroll,
							(int)(Message & 0x0000000f));
				}
				return PalmSecureConstant.JAVA_BioAPI_OK;
			}

			String key = "";
			if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_MOVING) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_MOVING;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_NO_HANDS) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_NO_HANDS;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_LESSINFO) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_LESSINFO;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_FAR) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_FAR;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_NEAR) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_NEAR;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_CAPTURING) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_CAPTURING;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_PHASE_END) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_PHASE_END;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_RIGHT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_RIGHT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_LEFT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_LEFT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_DOWN) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_DOWN;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_UP) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_START;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_PITCH_DOWN) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_PITCH_DOWN;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_PITCH_UP) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_PITCH_UP;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_ROLL_RIGHT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_ROLL_RIGHT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_ROLL_LEFT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_ROLL_LEFT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_YAW_RIGHT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_YAW_RIGHT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_YAW_LEFT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_YAW_LEFT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_ROUND) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_ROUND;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_ADJUST_LIGHT) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_ADJUST_LIGHT;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_ADJUST_NG) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_ADJUST_NG;

			} else if (Message == PalmSecureConstant.JAVA_PvAPI_NOTIFY_CAP_GUID_BADIMAGE) {
				key = PsFileAccessorLang.Guidance_NOTIFY_CAP_GUID_BADIMAGE;

			} else {
				return PalmSecureConstant.JAVA_BioAPI_OK;
			}

			frame.Ps_Sample_Apl_Java_NotifyGuidance(key, false);
	    }

		return PalmSecureConstant.JAVA_BioAPI_OK;

	}
}

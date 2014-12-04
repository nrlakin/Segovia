/*
 *	PsFileAccessor.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.xml;

import java.io.File;
import java.util.HashMap;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;

public abstract class PsFileAccessor {

	private static final String  NEW_LINE_CODE = System.getProperty("line.separator");

	protected void ReadXML(HashMap<String, String> xmlData, String fileName) throws Exception {

		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		DocumentBuilder builder = factory.newDocumentBuilder();

		File file = new File(fileName);
		Document doc = builder.parse(file);

		Element root = doc.getDocumentElement();
		ParseXML(xmlData, root, "");
	}

	private static void ParseXML(HashMap<String, String> xmlData, Element parent, String key) {

		NodeList children = parent.getChildNodes();

		for( int i = 0; i < children.getLength(); i++ ) {
			Node child = children.item(i);
			if ( child instanceof Element ) {
				Element childElement = (Element)child;
				if ( key.compareTo("") == 0 ) {
					ParseXML(xmlData, childElement, childElement.getTagName());
				} else {
					ParseXML(xmlData, childElement, key + "." + childElement.getTagName());
				}
			}
			if ( children.getLength() == 1 ) {
				if ( child instanceof Text ) {
					xmlData.put(key, child.getTextContent().replace("\\r\\n", NEW_LINE_CODE));
				}
			}
		}
	}


}

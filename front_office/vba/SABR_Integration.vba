'================================================================================
' SABR Microservice Integration Wrapper for Excel Front-Office
' Requires Reference to "Microsoft XML, v6.0" (MSXML2.ServerXMLHTTP60)
'================================================================================
Option Explicit

Private Const API_URL As String = "http://localhost:8000/api/v1/calibrate_sabr"

Public Function CalibrateSABR(ByVal ForwardRate As Double, _
                              ByVal TimeToMaturity As Double, _
                              ByVal StrikesRange As Range, _
                              ByVal MarketVolsRange As Range, _
                              Optional ByVal Beta As Double = 0.5, _
                              Optional ByVal Shift As Double = 0.0) As Variant
    
    Dim http As Object
    Dim jsonPayload As String
    Dim responseText As String
    Dim i As Integer
    Dim strikesStr As String
    Dim volsStr As String
    
    ' Extract data from Excel Ranges and format as JSON arrays
    strikesStr = "["
    volsStr = "["
    For i = 1 To StrikesRange.Cells.Count
        strikesStr = strikesStr & StrikesRange.Cells(i).Value
        volsStr = volsStr & MarketVolsRange.Cells(i).Value
        If i < StrikesRange.Cells.Count Then
            strikesStr = strikesStr & ", "
            volsStr = volsStr & ", "
        End If
    Next i
    strikesStr = strikesStr & "]"
    volsStr = volsStr & "]"
    
    ' Construct JSON Payload manually (in a real production environment, use a JSON parser library like VBA-JSON)
    jsonPayload = "{""forward_rate"": " & ForwardRate & ", " & _
                  """time_to_maturity"": " & TimeToMaturity & ", " & _
                  """strikes"": " & strikesStr & ", " & _
                  """market_vols"": " & volsStr & ", " & _
                  """beta"": " & Beta & ", " & _
                  """shift"": " & Shift & "}"
                  
    ' Send HTTP POST Request
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    
    On Error GoTo ErrorHandler
    http.Open "POST", API_URL, False
    http.setRequestHeader "Content-Type", "application/json"
    http.send jsonPayload
    
    If http.Status = 200 Then
        responseText = http.responseText
        ' Parse the response text. For simplicity in this scaffold, 
        ' we assume standard string manipulation to extract values.
        ' Expected JSON: {"alpha": 0.05, "beta": 0.5, "rho": -0.3, "nu": 0.4, "shift": 0.0, "mse": 0.0001, "status": "SUCCESS"}
        
        Dim outputArray(1 To 6, 1 To 2) As Variant
        outputArray(1, 1) = "Alpha": outputArray(1, 2) = ExtractJsonValue(responseText, "alpha")
        outputArray(2, 1) = "Beta": outputArray(2, 2) = ExtractJsonValue(responseText, "beta")
        outputArray(3, 1) = "Rho": outputArray(3, 2) = ExtractJsonValue(responseText, "rho")
        outputArray(4, 1) = "Nu": outputArray(4, 2) = ExtractJsonValue(responseText, "nu")
        outputArray(5, 1) = "Shift": outputArray(5, 2) = ExtractJsonValue(responseText, "shift")
        outputArray(6, 1) = "MSE": outputArray(6, 2) = ExtractJsonValue(responseText, "mse")
        
        CalibrateSABR = outputArray
    Else
        CalibrateSABR = "HTTP Error: " & http.Status & " - " & http.statusText
    End If
    
    Set http = Nothing
    Exit Function

ErrorHandler:
    CalibrateSABR = "Connection Error: Please ensure the Python API is running on " & API_URL
    Set http = Nothing
End Function

Private Function ExtractJsonValue(ByVal json As String, ByVal key As String) As Double
    ' A rudimentary JSON extractor for VBA scaffolding. 
    ' Finds the key and extracts the numeric value following the colon.
    Dim startPos As Long
    Dim endPos As Long
    Dim tempStr As String
    
    startPos = InStr(1, json, """" & key & """")
    If startPos > 0 Then
        startPos = InStr(startPos, json, ":") + 1
        endPos = InStr(startPos, json, ",")
        If endPos = 0 Then endPos = InStr(startPos, json, "}")
        
        tempStr = Trim(Mid(json, startPos, endPos - startPos))
        ExtractJsonValue = Val(tempStr)
    Else
        ExtractJsonValue = 0
    End If
End Function

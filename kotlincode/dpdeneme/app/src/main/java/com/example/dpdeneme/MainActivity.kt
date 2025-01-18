package com.example.dpdeneme

import android.annotation.SuppressLint
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import org.json.JSONObject
import java.io.IOException
import java.util.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.content.Context

class MainActivity : AppCompatActivity() {

    private lateinit var recordButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var text: TextView
    private val SPEECH_REQUEST_CODE = 0

    companion object {
        const val FLASK_API_URL = "http://192.168.1.242:5000/classify" // Flask API adresi
    }

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recordButton = findViewById(R.id.recordButton)
        resultTextView = findViewById(R.id.resultTextView)
         text = findViewById(R.id.textView2)

        // Butona tıklandığında ses algılamayı başlat
        recordButton.setOnClickListener {
            startSpeechToText()
        }
    }



    private fun startSpeechToText() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)

        intent.putExtra("android.speech.extra.SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS", 2500) // 5 saniye
        intent.putExtra("android.speech.extra.SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS", 2500) // 5 saniye
        intent.putExtra("android.speech.extra.SPEECH_INPUT_MINIMUM_LENGTH_MILLIS", 1500) // 1.5 saniye


        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US)

        try {
            startActivityForResult(intent, SPEECH_REQUEST_CODE)
        } catch (e: Exception) {
            e.printStackTrace()
            resultTextView.text = "Error starting speech recognizer: ${e.message}"
        }
    }

    var spokenText = ""
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == SPEECH_REQUEST_CODE && resultCode == RESULT_OK) {
            val results = data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            spokenText = results?.get(0) ?: ""
            resultTextView.text = "You said: $spokenText"

            // Python API'ye gönder
            classifyText(spokenText)
        }
    }

    private fun classifyText(inputText: String) {
        val client = OkHttpClient()

        // JSON verisini oluştur
        val json = JSONObject()
        json.put("text", inputText)

        val requestBody = RequestBody.create(
            "application/json".toMediaTypeOrNull(),
            json.toString()
        )

        val request = Request.Builder()
            .url(FLASK_API_URL)
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
                runOnUiThread {
                    resultTextView.text = "Error connecting to API: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (it.isSuccessful) {
                        val responseData = it.body?.string()
                        if (responseData != null) {
                            val jsonResponse = JSONObject(responseData)
                            val prediction = jsonResponse.getString("prediction")
                            val confidence = jsonResponse.getDouble("confidence")

                            // Flask API cevabına göre işlem yap
                            runOnUiThread {
                                if (prediction == "fraud") {
                                    resultTextView.text = "⚠ Fraud detected! Confidence: $confidence"
                                    text.text = spokenText
                                    vibratePhone()
                                    Toast.makeText(
                                        this@MainActivity,
                                        "Fraud detected!",
                                        Toast.LENGTH_LONG
                                    ).show()

                                } else {
                                    resultTextView.text = "✅ Normal speech detected. Confidence: $confidence"
                                    text.text = spokenText
                                    Toast.makeText(
                                        this@MainActivity,
                                        "Speech is normal.",
                                        Toast.LENGTH_LONG
                                    ).show()
                                }
                            }
                        }
                    } else {
                        runOnUiThread {
                            resultTextView.text = "Error: ${it.code}"
                        }
                    }
                }
            }

        })
    }

    private fun vibratePhone(duration: Long = 5000L) {
        val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(duration)
        }
    }


}
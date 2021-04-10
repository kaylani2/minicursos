const int DELAY_TIME = 1000; //ms
const int WEIGHT_PINS [6] = {A0, A1, A2, A3, A4, A5};
const int WEIGHT_OUTPUTS [6] = {3, 5, 6, 9, 10, 11}; //PWM
const int INPUT_A = 2;
const int INPUT_B = 4;
const int INPUT_SELECTOR = 7;
const int NN_OUTPUT = 12;
const int OUTPUT_THRESHOLD = 70;
int weightArray [6] = {0, 0, 0, 0, 0, 0};

void setup() {
  Serial.begin (9600);
  pinMode (INPUT_A, INPUT_PULLUP);
  pinMode (INPUT_B, INPUT_PULLUP);
  pinMode (INPUT_SELECTOR, INPUT_PULLUP);
  pinMode (NN_OUTPUT, OUTPUT);
  for (int index = 0; index < 6; index++)
    pinMode (WEIGHT_OUTPUTS [index], OUTPUT);
}


void loop() {
  int output = 0;
  Serial.println (!digitalRead (INPUT_A));
  Serial.println (!digitalRead (INPUT_B));

  for (int index = 0; index < 6; index++) {
    float weight = analogRead (WEIGHT_PINS [index]);
    weight = map (weight, 0, 1023, 0, 100);
    weightArray [index] = weight;
    Serial.print ("Peso ");
    Serial.print (index);
    Serial.print (": ");
    Serial.println (weight);
  }

  if (digitalRead (INPUT_SELECTOR) == LOW) {// Um neurônio
    Serial.println ("Rede neural rasa!");
    output = weightArray [0] * !digitalRead (INPUT_A) + weightArray [1] * !digitalRead (INPUT_B); // passar uma relu?
  }
  else { // Três neurônios
    Serial.println ("Rede neural profunda!");
    int firstNeuronActivation = weightArray [0] * !digitalRead (INPUT_A) + weightArray [1] * !digitalRead (INPUT_B);
    int secondNeuronActivation = weightArray [2] * !digitalRead (INPUT_A) + weightArray [3] * !digitalRead (INPUT_B);
    output = weightArray [4] * firstNeuronActivation + weightArray [5] * secondNeuronActivation;
  }
  Serial.println (output);
  (output >= OUTPUT_THRESHOLD) ? digitalWrite (NN_OUTPUT, HIGH) : digitalWrite (NN_OUTPUT, LOW);
  delay (DELAY_TIME);
}

package at.medunigraz.imi.bst.n2c2.nn;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import at.medunigraz.imi.bst.n2c2.classifier.Classifier;
import at.medunigraz.imi.bst.n2c2.model.Criterion;
import at.medunigraz.imi.bst.n2c2.model.Eligibility;
import at.medunigraz.imi.bst.n2c2.model.Patient;

/**
 * BI-LSTM classifier for n2c2 task 2018 refactored from dl4j examples.
 * 
 * @author Markus
 *
 */
public class BILSTMClassifier implements Classifier {

	// size of mini-batch for training
	private int miniBatchSize = 10;

	// length for truncated backpropagation through time
	private int tbpttLength = 10;

	// total number of training epochs
	private int nEpochs = 1000;

	// define initial time series length
	private int truncateLength = 64;

	// Google word vector size
	int vectorSize = 300;

	// accessing Google word vectors
	private WordVectors wordVectors;

	// training data
	private List<Patient> patientExamples;

	// tokenizer logic
	private TokenizerFactory tokenizerFactory;

	// multi layer network
	private MultiLayerNetwork net;

	// location of precalculated vectors
	public static final String WORD_VECTORS_PATH = "C:/Users/Markus/Downloads/GoogleNews-vectors-negative300.bin.gz";

	public BILSTMClassifier(List<Patient> examples) {

		this.patientExamples = examples;

		initializeTokenizer();
		initializeTruncateLength();
		initializeNetwork();
	}

	private void initializeTokenizer() {
		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
	}

	private void initializeNetwork() {

		// initialize network
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().updater(Updater.ADAM).adamMeanDecay(0.9)
				.adamVarDecay(0.999).regularization(true).l2(1e-5).weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).learningRate(2e-2).list()
				.layer(0,
						new GravesLSTM.Builder().nIn(vectorSize).nOut(truncateLength).activation(Activation.TANH)
								.build())
				.layer(1,
						new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
								.lossFunction(LossFunctions.LossFunction.MCXENT).nIn(truncateLength).nOut(2).build())
				.pretrain(false).backprop(true).build();

		this.net = new MultiLayerNetwork(conf);
		this.net.init();
		this.net.setListeners(new ScoreIterationListener(1));
	}

	/**
	 * Get longest token sequence of all patients with respect to existing word
	 * vector out of Google corpus.
	 * 
	 */
	private void initializeTruncateLength() {

		List<List<String>> allTokens = new ArrayList<>(patientExamples.size());
		int maxLength = 0;
		for (Patient patient : patientExamples) {
			String narrative = patient.getText();
			List<String> tokens = tokenizerFactory.create(narrative).getTokens();
			List<String> tokensFiltered = new ArrayList<>();
			for (String token : tokens) {
				if (wordVectors.hasWord(token))
					tokensFiltered.add(token);
			}
			allTokens.add(tokensFiltered);
			maxLength = Math.max(maxLength, tokensFiltered.size());
		}
		this.truncateLength = maxLength;
	}

	@Override
	public void train(List<Patient> examples) {

		// start training
		try {
			WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
			N2c2PatientIterator train = new N2c2PatientIterator(examples, wordVectors, miniBatchSize, truncateLength);

			System.out.println("Starting training");
			for (int i = 0; i < nEpochs; i++) {
				net.fit(train);
				train.reset();
				System.out.println("Epoch " + i + " complete. Starting evaluation:");

				// run evaluation on training set (should be test set)
				Evaluation evaluation = new Evaluation();
				while (train.hasNext()) {
					DataSet t = train.next();
					INDArray features = t.getFeatureMatrix();
					INDArray lables = t.getLabels();
					INDArray inMask = t.getFeaturesMaskArray();
					INDArray outMask = t.getLabelsMaskArray();
					INDArray predicted = net.output(features, false, inMask, outMask);

					evaluation.evalTimeSeries(lables, predicted, outMask);
				}
				train.reset();
				System.out.println(evaluation.stats());
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public Eligibility predict(Patient p) {
		return null;
	}

	@Override
	public Eligibility predict(Patient p, Criterion c) {
		return null;
	}

	@Override
	public List<Patient> predict(List<Patient> patientList) {
		return null;
	}
}

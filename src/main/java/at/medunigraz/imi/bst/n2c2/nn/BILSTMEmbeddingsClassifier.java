package at.medunigraz.imi.bst.n2c2.nn;

import at.medunigraz.imi.bst.n2c2.model.Criterion;
import at.medunigraz.imi.bst.n2c2.model.Patient;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.*;

public class BILSTMEmbeddingsClassifier extends BaseNNClassifier {

	private static final Logger LOG = LogManager.getLogger();

	private static final File WORD_VECTORS_FILE = new File(BILSTMC3GClassifier.class.getClassLoader().getResource("vectors.tsv").getFile());
	private static final WordVectors WORD_VECTORS = WordVectorSerializer.loadStaticModel(WORD_VECTORS_FILE);


	public void initializeNetworkFromFile(String pathToModel) {
		// settings for memory management:
		// https://deeplearning4j.org/workspaces

		Nd4j.getMemoryManager().setAutoGcWindow(10000);
		// Nd4j.getMemoryManager().togglePeriodicGc(false);

		// instantiating generator
//		fullSetIterator = new EmbeddingsIterator();

		Nd4j.getRandom().setSeed(12345);

		super.initializeNetworkFromFile(pathToModel);
	}

    protected void initializeNetwork() {
		initializeTruncateLength();
	    initializeNetworkBinaryMultiLabelDeep();
    }

	/**
	 * SIGMOID activation and XENT loss function for binary multi-label
	 * classification.
	 */
	protected void initializeNetworkBinaryMultiLabelDeep() {

		// settings for memory management:
		// https://deeplearning4j.org/workspaces

		Nd4j.getMemoryManager().setAutoGcWindow(10000);
		// Nd4j.getMemoryManager().togglePeriodicGc(false);

		fullSetIterator = new EmbeddingsIterator(patientExamples, WORD_VECTORS, BATCH_SIZE, truncateLength);
		vectorSize = ((EmbeddingsIterator)fullSetIterator).vectorSize;
		//truncateLength = ((EmbeddingsIterator)fullSetIterator).maxSentences;

		int nOutFF = 150;
		int lstmLayerSize = 128;
		double l2Regulization = 0.01;
		double adaGradCore = 0.04;
		double adaGradDense = 0.01;
		double adaGradGraves = 0.008;

		// seed for reproducibility
		final int seed = 12345;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.updater(AdaGrad.builder().learningRate(adaGradCore).build()).regularization(true).l2(l2Regulization)
				.weightInit(WeightInit.XAVIER).gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).trainingWorkspaceMode(WorkspaceMode.SINGLE)
				.inferenceWorkspaceMode(WorkspaceMode.SINGLE).list()

				.layer(0, new DenseLayer.Builder().activation(Activation.RELU).nIn(vectorSize).nOut(nOutFF)
						.weightInit(WeightInit.RELU).updater(AdaGrad.builder().learningRate(adaGradDense).build())
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10).build())

				.layer(1, new DenseLayer.Builder().activation(Activation.RELU).nIn(nOutFF).nOut(nOutFF)
						.weightInit(WeightInit.RELU).updater(AdaGrad.builder().learningRate(adaGradDense).build())
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10).build())

				.layer(2, new DenseLayer.Builder().activation(Activation.RELU).nIn(nOutFF).nOut(nOutFF)
						.weightInit(WeightInit.RELU).updater(AdaGrad.builder().learningRate(adaGradDense).build())
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10).build())

				.layer(3,
						new GravesBidirectionalLSTM.Builder().nIn(nOutFF).nOut(lstmLayerSize)
								.updater(AdaGrad.builder().learningRate(adaGradGraves).build())
								.activation(Activation.SOFTSIGN).build())

				.layer(4,
						new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
								.updater(AdaGrad.builder().learningRate(adaGradGraves).build())
								.activation(Activation.SOFTSIGN).build())

				.layer(5, new RnnOutputLayer.Builder().activation(Activation.SIGMOID)
						.lossFunction(LossFunctions.LossFunction.XENT).nIn(lstmLayerSize).nOut(Criterion.classifiableValues().length).build())

				.inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
				.inputPreProcessor(3, new FeedForwardToRnnPreProcessor()).pretrain(false).backprop(true).build();

		// .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)

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

		// type coverage
		Set<String> corpusTypes = new HashSet<String>();
		Set<String> matchedTypes = new HashSet<String>();

		// token coverage
		int filteredSum = 0;
		int tokenSum = 0;

		List<List<String>> allTokens = new ArrayList<>(patientExamples.size());
		int maxLength = 0;

		for (Patient patient : patientExamples) {
			String narrative = patient.getText();
//			String cleaned = narrative.replaceAll("[\r\n]+", " ").replaceAll("\\s+", " ");
			List<String> tokens = DataUtilities.getTokens(narrative);	// XXX Different than LSTMClassifier
			tokenSum += tokens.size();

			List<String> tokensFiltered = new ArrayList<>();
			for (String token : tokens) {
				corpusTypes.add(token);
				if (WORD_VECTORS.hasWord(token)) {
					tokensFiltered.add(token);
					matchedTypes.add(token);
				} else {
					LOG.info("Word2vec representation missing:\t" + token);
				}
			}
			allTokens.add(tokensFiltered);
			filteredSum += tokensFiltered.size();

			maxLength = Math.max(maxLength, tokensFiltered.size());
		}

		LOG.info("Matched " + matchedTypes.size() + " types out of " + corpusTypes.size());
		LOG.info("Matched " + filteredSum + " tokens out of " + tokenSum);

		this.truncateLength = maxLength;
	}

	protected String getModelName() {
		return "BLSTM_EMB";
	}

	/**
	 * Load features from narrative.
	 *
	 * @param reviewContents
	 *            Narrative content.
	 * @param maxLength
	 *            Maximum length of token series length.
	 * @return Time series feature presentation of narrative.
	 */
	protected INDArray loadFeaturesForNarrative(String reviewContents, int maxLength) {

		List<String> tokens = DataUtilities.getTokens(reviewContents);	// XXX Different than BILSTMC3GClassifier
		List<String> tokensFiltered = new ArrayList<>();
		for (String t : tokens) {
			if (WORD_VECTORS.hasWord(t))
				tokensFiltered.add(t);
		}
		int outputLength = Math.min(maxLength, tokensFiltered.size());

		INDArray features = Nd4j.create(1, vectorSize, outputLength);

		for (int j = 0; j < tokensFiltered.size() && j < maxLength; j++) {
			String token = tokensFiltered.get(j);
			INDArray vector = WORD_VECTORS.getWordVectorMatrix(token);
			features.put(new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j) },
					vector);
		}
		return features;
	}


}

/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

static constexpr int NUM_PARTICLES = 100;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /* Set the number of particles */
  num_particles = NUM_PARTICLES;

  std::default_random_engine gen;

  std::normal_distribution<double> x_dist{x, std[0]};
  std::normal_distribution<double> y_dist{y, std[1]};
  std::normal_distribution<double> theta_dist{theta, std[2]};

  /**
   * Initialize all particles to first position
   * (based on estimates of x, y, theta and their uncertainties
   * from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  Particle particle{};

  for(auto i = 0; i < num_particles; ++i)
  {
    particle.id = i;
    particle.x = x_dist(gen);
    particle.y = y_dist(gen);
    particle.theta = theta_dist(gen);
    particle.weight = 1.0 / NUM_PARTICLES;

    particles.push_back(particle);
	weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void update_position(double velocity, double yaw_rate, double delta_t, Particle& particle)
{ 
  static constexpr double min_yaw_rate = 0.001;

  if (std::fabs(yaw_rate) > min_yaw_rate)
  {
    particle.x += (velocity / yaw_rate)*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
    particle.y += (velocity / yaw_rate)*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
    particle.theta += yaw_rate*delta_t;
  }
  else
  {
    particle.x += velocity * delta_t * cos(particle.theta);
    particle.y += velocity * delta_t * sin(particle.theta);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate)
{
  std::default_random_engine gen;

  auto x_dist = [&std_pos](){return std::normal_distribution<double>{0.0, std_pos[0]};};
  auto y_dist = [&std_pos](){return std::normal_distribution<double>{0.0, std_pos[1]};};
  auto theta_dist = [&std_pos](){return std::normal_distribution<double>{0.0, std_pos[2]};};

  std::for_each(particles.begin(), particles.end(), [&](Particle& particle)
  {
    // Add measurements
    update_position(velocity, yaw_rate, delta_t, particle);

    // Add random Gaussian noise
    particle.x += x_dist()(gen);
    particle.y += y_dist()(gen);
    particle.theta += theta_dist()(gen);
  });

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
	std::vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */

  const auto get_closest_predicted_measurement = [&predicted](const LandmarkObs& observation){
	auto min_distance = std::numeric_limits<double>::max();
	LandmarkObs closest_observation{};

	for (const auto& predicted_measurement : predicted)
	{
		const auto distance = dist(predicted_measurement.x, predicted_measurement.y, observation.x, observation.y);
		if (distance < min_distance)
		{
			min_distance = distance;
			closest_observation = predicted_measurement;
		}
	}

	return closest_observation;
  };

  for(auto& observation : observations)
  {
    observation.id = get_closest_predicted_measurement(observation).id;
  }
}

std::vector<LandmarkObs> transform_observation(double x_translation, double y_translation,
	                                           double theta, const std::vector<LandmarkObs>& input)
{
	/**
	* Transform position from VEHICLE'S to MAP'S coordinate system
	*/

	std::vector<LandmarkObs> transformed_observations;

	std::transform(input.begin(), input.end(), std::back_inserter(transformed_observations),
		[&x_translation, &y_translation, &theta](const LandmarkObs& vehicle_obs) {

		LandmarkObs transformed_observation{};
		transformed_observation.x = x_translation + cos(theta) * vehicle_obs.x - sin(theta) * vehicle_obs.y;
		transformed_observation.y = y_translation + sin(theta) * vehicle_obs.x + cos(theta) * vehicle_obs.y;
		transformed_observation.id = vehicle_obs.id;

		return transformed_observation;
	});


	return transformed_observations;
}

std::vector<LandmarkObs> select_landmarks(const std::vector<Map::single_landmark_s>& landmarks,
										  double x_ref,
										  double y_ref,
										  double sensor_range)
{
	/**
	* Filter out landmarks which are not within sensor range
	*/
	std::vector<LandmarkObs> filtered_landmarks;

	for (auto landmark : landmarks)
	{
		const auto landmark_dist = dist(x_ref, y_ref, landmark.x_f, landmark.y_f);
		if (landmark_dist < sensor_range) 
		{
			filtered_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
		}
	}

	return filtered_landmarks;
}

double gaussian(double sigma_x, double sigma_y, double delta_x, double delta_y)
{
	//  Multi-variate Gaussian
	const auto exponent = pow(delta_x, 2) / (2 * pow(sigma_x, 2)) + pow(delta_y, 2) / (2 * pow(sigma_y, 2));
	return exp(-exponent) / (2.0 * M_PI * sigma_x * sigma_y);
}

double ParticleFilter::calculateParticleWeight(const Particle& particle, double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations,
	const Map &map_landmarks)
{
	double weight = 1.0;

	// Select landmarks within defined range
	auto predictions = select_landmarks(map_landmarks.landmark_list, particle.x, particle.y, sensor_range);

	// Find observations' coordinates in MAP'S coordinate system
	auto map_observations = transform_observation(particle.x, particle.y, particle.theta, observations);

	// Find the predicted measurement that is closest to each observed measurement
	dataAssociation(predictions, map_observations);

	// Calculate particle's weight:
	Map::single_landmark_s landmark{};
	for (const auto& map_observation : map_observations) {

		landmark = map_landmarks.landmark_list.at(map_observation.id - 1);
		weight *= gaussian(std_landmark[0], std_landmark[1], (map_observation.x - landmark.x_f), (map_observation.y - landmark.y_f));
	}

	return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

	for (size_t i = 0; i < particles.size(); ++i) 
	{
		particles[i].weight = calculateParticleWeight(particles[i], sensor_range, std_landmark, observations, map_landmarks);
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
  /**
   * Wheel resampling implementation
   * Resample particles with replacement with probability proportional 
   * to their weight. 
   */
	std::vector<Particle> resampled_particles(num_particles);

	int index = rand() % num_particles;

	double max_weight = *max_element(weights.begin(), weights.end());

	double beta = 0.0;
	double beta_random_part = 0.0;
	for (auto& resampled_particle : resampled_particles)
	{
		beta_random_part = ((double)rand() / (RAND_MAX)) + 1;
		beta += beta_random_part*(2.0 * max_weight);
		while (weights[index] < beta) 
		{
			beta -= weights[index];
			index = (++index) % num_particles;
		}
		resampled_particle = particles[index];
	}

	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
	std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
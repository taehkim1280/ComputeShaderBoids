extends Node2D

var NUM_BOIDS := 5000
var boid_pos := []
var boid_vel := []
var boid_speed := []

var IMAGE_SIZE := int(ceil(sqrt(NUM_BOIDS)))
var boid_data : Image
var boid_data_texture : ImageTexture

@export_category("Boid Settings")
@export_range(0, 50) var friend_radius := 10.0
@export_range(0, 50) var avoid_radius := 5.0
@export_range(0,100) var min_vel := 50.0
@export_range(0,100) var max_vel := 100.0
@export_range(0,100) var alignment_factor := 10.0
@export_range(0,100) var cohesion_factor := 1.0
@export_range(0,100) var separation_factor := 20.0

# GPU Variables
var SIMULATE_GPU = true
var rd : RenderingDevice
var boid_compute_shader : RID
var boid_pipeline : RID
var bindings : Array
var uniform_set : RID

var boid_pos_buffer : RID
var boid_vel_buffer : RID
var boid_speed_buffer : RID
var params_buffer: RID
var params_uniform : RDUniform
var boid_data_buffer : RID


func _generate_boids():
	for i in NUM_BOIDS:
		boid_pos.append(Vector2(randf() * get_viewport_rect().size.x, randf() * get_viewport_rect().size.y))
		boid_vel.append(Vector2(randf_range(-1.0, 1.0) * randf_range(min_vel, max_vel), randf_range(-1.0, 1.0) * randf_range(min_vel, max_vel)))
		boid_speed.append(randf_range(min_vel, max_vel))


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	_generate_boids()


	boid_data = Image.create(IMAGE_SIZE, IMAGE_SIZE, false, Image.FORMAT_RGBAH)
	boid_data_texture = ImageTexture.create_from_image(boid_data)

	$BoidParticles.amount = NUM_BOIDS
	$BoidParticles.process_material.set_shader_parameter("boid_data", boid_data_texture)

	if SIMULATE_GPU:
		_setup_compute_shader()
		_update_boids_gpu(0)

func _update_data_texture():
	if SIMULATE_GPU:
		# //////////////////////////////////////////////////////////////////////////////////////
		# //////////////     GET DATA FROM GPU TO SEND TO PARTICLE SHADER      //////////////////
		# //////////////////////////////////////////////////////////////////////////////////////
		var boid_data_image_data := rd.texture_get_data(boid_data_buffer, 0)
		boid_data.set_data(IMAGE_SIZE, IMAGE_SIZE, false, Image.FORMAT_RGBAH, boid_data_image_data)
	else:
		for i in NUM_BOIDS:
			var pixel_pos = Vector2(int(i / IMAGE_SIZE), int(i % IMAGE_SIZE))
			boid_data.set_pixel(pixel_pos.x, pixel_pos.y, Color(boid_pos[i].x, boid_pos[i].y, boid_vel[i].angle(), 0))

	boid_data_texture.update(boid_data)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	get_window().title = "Boids: " + str(NUM_BOIDS) + " / FPS: " + str(Engine.get_frames_per_second())

	if SIMULATE_GPU:
		_sync_boids_gpu()
	else:
		_update_boid_cpu(delta)

	_update_data_texture()

	if SIMULATE_GPU:
		_update_boids_gpu(delta)

func _update_boid_cpu(delta):
	for i in NUM_BOIDS:
		var my_pos = boid_pos[i]
		var my_vel = boid_vel[i]
		var avg_vel = Vector2.ZERO
		var midpoint = Vector2.ZERO
		var separation_vec = Vector2.ZERO
		var num_friends = 0
		var num_avoids = 0
		for j in NUM_BOIDS:
			if i != j:
				var other_pos = boid_pos[j]
				var other_vel = boid_vel[j]
				var dist = my_pos.distance_to(other_pos)
				if(dist < friend_radius):
					num_friends += 1
					avg_vel += other_vel
					midpoint += other_pos
					if(dist < avoid_radius):
						num_avoids += 1
						separation_vec += my_pos - other_pos
					
		if(num_friends > 0):
			avg_vel /= num_friends
			my_vel += avg_vel.normalized() * alignment_factor
			
			midpoint /= num_friends
			my_vel += (midpoint - my_pos).normalized() * cohesion_factor
			
			if(num_avoids > 0):
				my_vel += separation_vec.normalized() * separation_factor
		
		var vel_mag = my_vel.length()
		vel_mag = clamp(vel_mag, min_vel, max_vel)
		my_vel = my_vel.normalized() * vel_mag

		# Time step
		my_pos += my_vel * delta

		# Wrap around screen
		my_pos = Vector2(wrapf(my_pos.x, 0, get_viewport_rect().size.x,),
						 wrapf(my_pos.y, 0, get_viewport_rect().size.y,))
		
		# Update state
		boid_pos[i] = my_pos
		boid_vel[i] = my_vel 


func _setup_compute_shader():

	# //////////////////////////////////////////////////////////
	# //////////////////     SETUP      ///////////////////////
	# //////////////////////////////////////////////////////////

	rd = RenderingServer.create_local_rendering_device()

	# load shader file
	var shader_file := load("res://compute_shaders/boid_simulation.glsl")
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()
	boid_compute_shader = rd.shader_create_from_spirv(shader_spirv)

	# set member with pipeline
	boid_pipeline = rd.compute_pipeline_create(boid_compute_shader)
	
	# //////////////////////////////////////////////////////////
	# //////////////////     INPUTS      ///////////////////////
	# //////////////////////////////////////////////////////////

	# pos buffer
	boid_pos_buffer = _generate_vec2_buffer(boid_pos)
	var boid_pos_uniform = _generate_uniform(boid_pos_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 0)
	# vel buffer
	boid_vel_buffer = _generate_vec2_buffer(boid_vel)
	var boid_vel_uniform = _generate_uniform(boid_vel_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 1)
	# speed buffer
	boid_speed_buffer = _generate_vec2_buffer(boid_speed)
	var boid_speed_uniform = _generate_uniform(boid_speed_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 2)
	# params buffer
	params_buffer = _generate_parameter_buffer(0)
	params_uniform = _generate_uniform(params_buffer, RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 3)

	######## boid_data buffer (compute shader output) ########
	# sets the texture format
	var fmt := RDTextureFormat.new()
	fmt.width = IMAGE_SIZE
	fmt.height = IMAGE_SIZE
	fmt.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	fmt.usage_bits = (RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT 
		| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT)
	# uses texture format to make texture containing boid_data image
	var view := RDTextureView.new()
	boid_data_buffer = rd.texture_create(fmt, view, [boid_data.get_data()])
	var boid_data_buffer_uniform = _generate_uniform(boid_data_buffer, RenderingDevice.UNIFORM_TYPE_IMAGE, 4)


	# set bindings for uniform set
	bindings = [boid_pos_uniform, 
				boid_vel_uniform, 
				boid_speed_uniform,
				params_uniform,
				boid_data_buffer_uniform]
	uniform_set = rd.uniform_set_create(bindings, boid_compute_shader, 0)

func _update_boids_gpu(delta):
	rd.free_rid(params_buffer)
	params_buffer = _generate_parameter_buffer(delta)
	params_uniform.clear_ids()
	params_uniform.add_id(params_buffer)
	uniform_set = rd.uniform_set_create(bindings, boid_compute_shader, 0)

	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, boid_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)

	# //////////////////////////////////////////////////////////
	# //////////////////     EXECUTE      ///////////////////////
	# //////////////////////////////////////////////////////////

	rd.compute_list_dispatch(compute_list, ceil(NUM_BOIDS/1024.), 1, 1)
	rd.compute_list_end()
	rd.submit()


func _sync_boids_gpu():
	rd.sync()

func _exit_tree():
	if SIMULATE_GPU:
		_sync_boids_gpu()
		rd.free_rid(uniform_set)
		rd.free_rid(boid_data_buffer)
		rd.free_rid(params_buffer)
		rd.free_rid(boid_pos_buffer)
		rd.free_rid(boid_vel_buffer)
		rd.free_rid(boid_speed_buffer)
		rd.free_rid(boid_pipeline)
		rd.free_rid(boid_compute_shader)
		rd.free()



# //////////////////////////////////////////////////////////////////////
# /////////////////////     HELPER FUNCTIONS    /////////////////////////
# //////////////////////////////////////////////////////////////////////

func _generate_vec2_buffer(data):
	var data_buffer_bytes := PackedVector2Array(data).to_byte_array()
	var data_buffer = rd.storage_buffer_create(data_buffer_bytes.size(), data_buffer_bytes)
	return data_buffer

func _generate_int_buffer(size):
	var data = []
	data.resize(size)
	var data_buffer_bytes = PackedInt32Array(data).to_byte_array()
	var data_buffer = rd.storage_buffer_create(data_buffer_bytes.size(), data_buffer_bytes)
	return data_buffer
	
func _generate_uniform(data_buffer, type, binding):
	var data_uniform = RDUniform.new()
	data_uniform.uniform_type = type
	data_uniform.binding = binding
	data_uniform.add_id(data_buffer)
	return data_uniform


func _generate_parameter_buffer(delta):
	var params_buffer_bytes : PackedByteArray = PackedFloat32Array(
		[NUM_BOIDS, 
		IMAGE_SIZE, 
		friend_radius,
		avoid_radius,
		min_vel, 
		max_vel,
		alignment_factor,
		cohesion_factor,
		separation_factor,
		get_viewport_rect().size.x,
		get_viewport_rect().size.y,
		delta]).to_byte_array()
	
	return rd.storage_buffer_create(params_buffer_bytes.size(), params_buffer_bytes)
